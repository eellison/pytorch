class GraphModule(torch.nn.Module):
    def forward(self, primals_4: f32[768], primals_14: f32[768], primals_20: f32[768], primals_30: f32[768], primals_36: f32[768], primals_46: f32[768], primals_52: f32[768], primals_62: f32[768], primals_68: f32[768], primals_78: f32[768], primals_84: f32[768], primals_94: f32[768], primals_100: f32[768], primals_110: f32[768], primals_116: f32[768], primals_126: f32[768], primals_132: f32[768], primals_142: f32[768], primals_148: f32[768], primals_158: f32[768], primals_164: f32[768], primals_174: f32[768], primals_180: f32[768], primals_190: f32[768], primals_196: f32[768], primals_200: f32[768], slice_2: i64[1, 128], mul_1: f32[64, 128, 768], gt: b8[64, 128, 768], view: f32[8192, 768], gt_1: b8[64, 12, 128, 128], view_11: f32[8192, 768], gt_2: b8[64, 128, 768], mul_9: f32[64, 128, 768], view_13: f32[8192, 768], view_15: f32[8192, 3072], gt_3: b8[64, 128, 768], mul_26: f32[64, 128, 768], view_17: f32[8192, 768], gt_4: b8[64, 12, 128, 128], view_28: f32[8192, 768], gt_5: b8[64, 128, 768], mul_32: f32[64, 128, 768], view_30: f32[8192, 768], view_32: f32[8192, 3072], gt_6: b8[64, 128, 768], mul_49: f32[64, 128, 768], view_34: f32[8192, 768], gt_7: b8[64, 12, 128, 128], view_45: f32[8192, 768], gt_8: b8[64, 128, 768], mul_55: f32[64, 128, 768], view_47: f32[8192, 768], view_49: f32[8192, 3072], gt_9: b8[64, 128, 768], mul_72: f32[64, 128, 768], view_51: f32[8192, 768], gt_10: b8[64, 12, 128, 128], view_62: f32[8192, 768], gt_11: b8[64, 128, 768], mul_78: f32[64, 128, 768], view_64: f32[8192, 768], view_66: f32[8192, 3072], gt_12: b8[64, 128, 768], mul_95: f32[64, 128, 768], view_68: f32[8192, 768], gt_13: b8[64, 12, 128, 128], view_79: f32[8192, 768], gt_14: b8[64, 128, 768], mul_101: f32[64, 128, 768], view_81: f32[8192, 768], view_83: f32[8192, 3072], gt_15: b8[64, 128, 768], mul_118: f32[64, 128, 768], view_85: f32[8192, 768], gt_16: b8[64, 12, 128, 128], view_96: f32[8192, 768], gt_17: b8[64, 128, 768], mul_124: f32[64, 128, 768], view_98: f32[8192, 768], view_100: f32[8192, 3072], gt_18: b8[64, 128, 768], mul_141: f32[64, 128, 768], view_102: f32[8192, 768], gt_19: b8[64, 12, 128, 128], view_113: f32[8192, 768], gt_20: b8[64, 128, 768], mul_147: f32[64, 128, 768], view_115: f32[8192, 768], view_117: f32[8192, 3072], gt_21: b8[64, 128, 768], mul_164: f32[64, 128, 768], view_119: f32[8192, 768], gt_22: b8[64, 12, 128, 128], view_130: f32[8192, 768], gt_23: b8[64, 128, 768], mul_170: f32[64, 128, 768], view_132: f32[8192, 768], view_134: f32[8192, 3072], gt_24: b8[64, 128, 768], mul_187: f32[64, 128, 768], view_136: f32[8192, 768], gt_25: b8[64, 12, 128, 128], view_147: f32[8192, 768], gt_26: b8[64, 128, 768], mul_193: f32[64, 128, 768], view_149: f32[8192, 768], view_151: f32[8192, 3072], gt_27: b8[64, 128, 768], mul_210: f32[64, 128, 768], view_153: f32[8192, 768], gt_28: b8[64, 12, 128, 128], view_164: f32[8192, 768], gt_29: b8[64, 128, 768], mul_216: f32[64, 128, 768], view_166: f32[8192, 768], view_168: f32[8192, 3072], gt_30: b8[64, 128, 768], mul_233: f32[64, 128, 768], view_170: f32[8192, 768], gt_31: b8[64, 12, 128, 128], view_181: f32[8192, 768], gt_32: b8[64, 128, 768], mul_239: f32[64, 128, 768], view_183: f32[8192, 768], view_185: f32[8192, 3072], gt_33: b8[64, 128, 768], mul_256: f32[64, 128, 768], view_187: f32[8192, 768], gt_34: b8[64, 12, 128, 128], view_198: f32[8192, 768], gt_35: b8[64, 128, 768], mul_262: f32[64, 128, 768], view_200: f32[8192, 768], view_202: f32[8192, 3072], gt_36: b8[64, 128, 768], mul_279: f32[64, 128, 768], view_204: f32[8192, 768], mul_294: f32[64, 128, 768], view_206: f32[8192, 768], sub_53: f32[8192, 30522], unsqueeze_2: i64[8192, 1], permute_134: f32[30522, 768], div_25: f32[64, 128, 1], add_175: f32[64, 128, 768], permute_138: f32[768, 768], div_26: f32[64, 128, 1], permute_142: f32[768, 3072], add_182: f32[64, 128, 3072], permute_146: f32[3072, 768], div_27: f32[64, 128, 1], permute_150: f32[768, 768], permute_155: f32[768, 128, 128], permute_156: f32[768, 64, 128], alias_83: f32[64, 12, 128, 128], permute_157: f32[768, 64, 128], permute_158: f32[768, 128, 64], permute_162: f32[768, 768], permute_167: f32[768, 768], permute_171: f32[768, 768], div_29: f32[64, 128, 1], permute_175: f32[768, 3072], add_193: f32[64, 128, 3072], permute_179: f32[3072, 768], div_30: f32[64, 128, 1], permute_183: f32[768, 768], permute_188: f32[768, 128, 128], permute_189: f32[768, 64, 128], alias_85: f32[64, 12, 128, 128], permute_190: f32[768, 64, 128], permute_191: f32[768, 128, 64], permute_195: f32[768, 768], permute_200: f32[768, 768], permute_204: f32[768, 768], div_32: f32[64, 128, 1], permute_208: f32[768, 3072], add_204: f32[64, 128, 3072], permute_212: f32[3072, 768], div_33: f32[64, 128, 1], permute_216: f32[768, 768], permute_221: f32[768, 128, 128], permute_222: f32[768, 64, 128], alias_87: f32[64, 12, 128, 128], permute_223: f32[768, 64, 128], permute_224: f32[768, 128, 64], permute_228: f32[768, 768], permute_233: f32[768, 768], permute_237: f32[768, 768], div_35: f32[64, 128, 1], permute_241: f32[768, 3072], add_215: f32[64, 128, 3072], permute_245: f32[3072, 768], div_36: f32[64, 128, 1], permute_249: f32[768, 768], permute_254: f32[768, 128, 128], permute_255: f32[768, 64, 128], alias_89: f32[64, 12, 128, 128], permute_256: f32[768, 64, 128], permute_257: f32[768, 128, 64], permute_261: f32[768, 768], permute_266: f32[768, 768], permute_270: f32[768, 768], div_38: f32[64, 128, 1], permute_274: f32[768, 3072], add_226: f32[64, 128, 3072], permute_278: f32[3072, 768], div_39: f32[64, 128, 1], permute_282: f32[768, 768], permute_287: f32[768, 128, 128], permute_288: f32[768, 64, 128], alias_91: f32[64, 12, 128, 128], permute_289: f32[768, 64, 128], permute_290: f32[768, 128, 64], permute_294: f32[768, 768], permute_299: f32[768, 768], permute_303: f32[768, 768], div_41: f32[64, 128, 1], permute_307: f32[768, 3072], add_237: f32[64, 128, 3072], permute_311: f32[3072, 768], div_42: f32[64, 128, 1], permute_315: f32[768, 768], permute_320: f32[768, 128, 128], permute_321: f32[768, 64, 128], alias_93: f32[64, 12, 128, 128], permute_322: f32[768, 64, 128], permute_323: f32[768, 128, 64], permute_327: f32[768, 768], permute_332: f32[768, 768], permute_336: f32[768, 768], div_44: f32[64, 128, 1], permute_340: f32[768, 3072], add_248: f32[64, 128, 3072], permute_344: f32[3072, 768], div_45: f32[64, 128, 1], permute_348: f32[768, 768], permute_353: f32[768, 128, 128], permute_354: f32[768, 64, 128], alias_95: f32[64, 12, 128, 128], permute_355: f32[768, 64, 128], permute_356: f32[768, 128, 64], permute_360: f32[768, 768], permute_365: f32[768, 768], permute_369: f32[768, 768], div_47: f32[64, 128, 1], permute_373: f32[768, 3072], add_259: f32[64, 128, 3072], permute_377: f32[3072, 768], div_48: f32[64, 128, 1], permute_381: f32[768, 768], permute_386: f32[768, 128, 128], permute_387: f32[768, 64, 128], alias_97: f32[64, 12, 128, 128], permute_388: f32[768, 64, 128], permute_389: f32[768, 128, 64], permute_393: f32[768, 768], permute_398: f32[768, 768], permute_402: f32[768, 768], div_50: f32[64, 128, 1], permute_406: f32[768, 3072], add_270: f32[64, 128, 3072], permute_410: f32[3072, 768], div_51: f32[64, 128, 1], permute_414: f32[768, 768], permute_419: f32[768, 128, 128], permute_420: f32[768, 64, 128], alias_99: f32[64, 12, 128, 128], permute_421: f32[768, 64, 128], permute_422: f32[768, 128, 64], permute_426: f32[768, 768], permute_431: f32[768, 768], permute_435: f32[768, 768], div_53: f32[64, 128, 1], permute_439: f32[768, 3072], add_281: f32[64, 128, 3072], permute_443: f32[3072, 768], div_54: f32[64, 128, 1], permute_447: f32[768, 768], permute_452: f32[768, 128, 128], permute_453: f32[768, 64, 128], alias_101: f32[64, 12, 128, 128], permute_454: f32[768, 64, 128], permute_455: f32[768, 128, 64], permute_459: f32[768, 768], permute_464: f32[768, 768], permute_468: f32[768, 768], div_56: f32[64, 128, 1], permute_472: f32[768, 3072], add_292: f32[64, 128, 3072], permute_476: f32[3072, 768], div_57: f32[64, 128, 1], permute_480: f32[768, 768], permute_485: f32[768, 128, 128], permute_486: f32[768, 64, 128], alias_103: f32[64, 12, 128, 128], permute_487: f32[768, 64, 128], permute_488: f32[768, 128, 64], permute_492: f32[768, 768], permute_497: f32[768, 768], permute_501: f32[768, 768], div_59: f32[64, 128, 1], permute_505: f32[768, 3072], add_303: f32[64, 128, 3072], permute_509: f32[3072, 768], div_60: f32[64, 128, 1], permute_513: f32[768, 768], permute_518: f32[768, 128, 128], permute_519: f32[768, 64, 128], alias_105: f32[64, 12, 128, 128], permute_520: f32[768, 64, 128], permute_521: f32[768, 128, 64], permute_525: f32[768, 768], permute_530: f32[768, 768], permute_534: f32[768, 768], div_62: f32[64, 128, 1], view_506: i64[128], view_509: i64[8192], tangents_1: f32[], tangents_2: f32[64, 128, 30522]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:980, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: i64[64, 128] = torch.ops.aten.expand.default(slice_2, [64, 128]);  slice_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1367, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        alias_76: f32[8192, 30522] = torch.ops.aten.alias.default(sub_53)
        alias_77: f32[8192, 30522] = torch.ops.aten.alias.default(alias_76);  alias_76 = None
        full: f32[] = torch.ops.aten.full.default([], 8192.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_78: f32[] = torch.ops.aten.alias.default(full);  full = None
        div_24: f32[] = torch.ops.aten.div.Tensor(tangents_1, alias_78);  tangents_1 = alias_78 = None
        zeros_like: f32[8192, 30522] = torch.ops.aten.zeros_like.default(sub_53, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sub_53 = None
        alias_79: f32[8192, 30522] = torch.ops.aten.alias.default(zeros_like);  zeros_like = None
        scatter: f32[8192, 30522] = torch.ops.aten.scatter.value(alias_79, 1, unsqueeze_2, -1.0);  alias_79 = unsqueeze_2 = None
        mul_296: f32[8192, 30522] = torch.ops.aten.mul.Tensor(scatter, div_24);  scatter = div_24 = None
        alias_80: f32[8192, 30522] = torch.ops.aten.alias.default(alias_77);  alias_77 = None
        alias_81: f32[8192, 30522] = torch.ops.aten.alias.default(alias_80);  alias_80 = None
        exp_26: f32[8192, 30522] = torch.ops.aten.exp.default(alias_81);  alias_81 = None
        sum_14: f32[8192, 1] = torch.ops.aten.sum.dim_IntList(mul_296, [1], True)
        mul_297: f32[8192, 30522] = torch.ops.aten.mul.Tensor(exp_26, sum_14);  exp_26 = sum_14 = None
        sub_54: f32[8192, 30522] = torch.ops.aten.sub.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        view_210: f32[64, 128, 30522] = torch.ops.aten.view.default(sub_54, [64, 128, 30522]);  sub_54 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1367, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        add_168: f32[64, 128, 30522] = torch.ops.aten.add.Tensor(tangents_2, view_210);  tangents_2 = view_210 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:693, code: hidden_states = self.decoder(hidden_states)
        view_211: f32[8192, 30522] = torch.ops.aten.view.default(add_168, [8192, 30522]);  add_168 = None
        mm: f32[8192, 768] = torch.ops.aten.mm.default(view_211, permute_134);  permute_134 = None
        permute_135: f32[30522, 8192] = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_1: f32[30522, 768] = torch.ops.aten.mm.default(permute_135, view_206);  permute_135 = view_206 = None
        permute_136: f32[768, 30522] = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_15: f32[1, 30522] = torch.ops.aten.sum.dim_IntList(view_211, [0], True);  view_211 = None
        view_212: f32[30522] = torch.ops.aten.view.default(sum_15, [30522]);  sum_15 = None
        view_213: f32[64, 128, 768] = torch.ops.aten.view.default(mm, [64, 128, 768]);  mm = None
        permute_137: f32[30522, 768] = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:673, code: hidden_states = self.LayerNorm(hidden_states)
        mul_299: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_213, primals_200);  primals_200 = None
        mul_300: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_299, 768)
        sum_16: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_299, mul_294);  mul_299 = None
        sum_17: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_294, sum_17);  sum_17 = None
        sub_56: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_300, sum_16);  mul_300 = sum_16 = None
        sub_57: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_56, mul_302);  sub_56 = mul_302 = None
        mul_303: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_25, sub_57);  div_25 = sub_57 = None
        mul_304: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_213, mul_294);  mul_294 = None
        sum_18: f32[768] = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        sum_19: f32[768] = torch.ops.aten.sum.dim_IntList(view_213, [0, 1]);  view_213 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_321: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_303, add_175);  mul_303 = add_175 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:671, code: hidden_states = self.dense(hidden_states)
        view_214: f32[8192, 768] = torch.ops.aten.view.default(mul_321, [8192, 768]);  mul_321 = None
        mm_2: f32[8192, 768] = torch.ops.aten.mm.default(view_214, permute_138);  permute_138 = None
        permute_139: f32[768, 8192] = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_3: f32[768, 768] = torch.ops.aten.mm.default(permute_139, view_204);  permute_139 = view_204 = None
        permute_140: f32[768, 768] = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_20: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        view_215: f32[768] = torch.ops.aten.view.default(sum_20, [768]);  sum_20 = None
        view_216: f32[64, 128, 768] = torch.ops.aten.view.default(mm_2, [64, 128, 768]);  mm_2 = None
        permute_141: f32[768, 768] = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_323: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_216, primals_196);  primals_196 = None
        mul_324: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_323, 768)
        sum_21: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
        mul_325: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_323, mul_279);  mul_323 = None
        sum_22: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
        mul_326: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_279, sum_22);  sum_22 = None
        sub_60: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_324, sum_21);  mul_324 = sum_21 = None
        sub_61: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_60, mul_326);  sub_60 = mul_326 = None
        mul_327: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_26, sub_61);  div_26 = sub_61 = None
        mul_328: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_216, mul_279);  mul_279 = None
        sum_23: f32[768] = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
        sum_24: f32[768] = torch.ops.aten.sum.dim_IntList(view_216, [0, 1]);  view_216 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_36, dtype = torch.float32);  gt_36 = None
        mul_329: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy, 1.1111111111111112);  _to_copy = None
        mul_330: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_327, mul_329);  mul_329 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_217: f32[8192, 768] = torch.ops.aten.view.default(mul_330, [8192, 768]);  mul_330 = None
        mm_4: f32[8192, 3072] = torch.ops.aten.mm.default(view_217, permute_142);  permute_142 = None
        permute_143: f32[768, 8192] = torch.ops.aten.permute.default(view_217, [1, 0])
        mm_5: f32[768, 3072] = torch.ops.aten.mm.default(permute_143, view_202);  permute_143 = view_202 = None
        permute_144: f32[3072, 768] = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_25: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_217, [0], True);  view_217 = None
        view_218: f32[768] = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
        view_219: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_4, [64, 128, 3072]);  mm_4 = None
        permute_145: f32[768, 3072] = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_347: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_219, add_182);  view_219 = add_182 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_220: f32[8192, 3072] = torch.ops.aten.view.default(mul_347, [8192, 3072]);  mul_347 = None
        mm_6: f32[8192, 768] = torch.ops.aten.mm.default(view_220, permute_146);  permute_146 = None
        permute_147: f32[3072, 8192] = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_7: f32[3072, 768] = torch.ops.aten.mm.default(permute_147, view_200);  permute_147 = view_200 = None
        permute_148: f32[768, 3072] = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_26: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
        view_221: f32[3072] = torch.ops.aten.view.default(sum_26, [3072]);  sum_26 = None
        view_222: f32[64, 128, 768] = torch.ops.aten.view.default(mm_6, [64, 128, 768]);  mm_6 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_183: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_327, view_222);  mul_327 = view_222 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_149: f32[3072, 768] = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_349: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_183, primals_190);  primals_190 = None
        mul_350: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_349, 768)
        sum_27: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_349, [2], True)
        mul_351: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_349, mul_262);  mul_349 = None
        sum_28: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
        mul_352: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_262, sum_28);  sum_28 = None
        sub_64: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_350, sum_27);  mul_350 = sum_27 = None
        sub_65: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_64, mul_352);  sub_64 = mul_352 = None
        mul_353: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_27, sub_65);  div_27 = sub_65 = None
        mul_354: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_183, mul_262);  mul_262 = None
        sum_29: f32[768] = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1]);  mul_354 = None
        sum_30: f32[768] = torch.ops.aten.sum.dim_IntList(add_183, [0, 1]);  add_183 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_1: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_35, dtype = torch.float32);  gt_35 = None
        mul_355: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_1, 1.1111111111111112);  _to_copy_1 = None
        mul_356: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_353, mul_355);  mul_355 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_223: f32[8192, 768] = torch.ops.aten.view.default(mul_356, [8192, 768]);  mul_356 = None
        mm_8: f32[8192, 768] = torch.ops.aten.mm.default(view_223, permute_150);  permute_150 = None
        permute_151: f32[768, 8192] = torch.ops.aten.permute.default(view_223, [1, 0])
        mm_9: f32[768, 768] = torch.ops.aten.mm.default(permute_151, view_198);  permute_151 = view_198 = None
        permute_152: f32[768, 768] = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_31: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
        view_224: f32[768] = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
        view_225: f32[64, 128, 768] = torch.ops.aten.view.default(mm_8, [64, 128, 768]);  mm_8 = None
        permute_153: f32[768, 768] = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_226: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_225, [64, 128, 12, 64]);  view_225 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_154: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_48: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        _unsafe_view_60: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_48, [768, 128, 64]);  clone_48 = None
        bmm_24: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_155, _unsafe_view_60);  permute_155 = None
        bmm_25: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_60, permute_156);  _unsafe_view_60 = permute_156 = None
        view_227: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_24, [64, 12, 128, 64]);  bmm_24 = None
        view_228: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_25, [64, 12, 128, 128]);  bmm_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_2: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_34, dtype = torch.float32);  gt_34 = None
        mul_357: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_2, 1.1111111111111112);  _to_copy_2 = None
        mul_358: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_228, mul_357);  view_228 = mul_357 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_359: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_358, alias_83);  mul_358 = None
        sum_32: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_359, [-1], True)
        mul_360: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_83, sum_32);  alias_83 = sum_32 = None
        sub_66: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_28: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_66, 8.0);  sub_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_229: f32[768, 128, 128] = torch.ops.aten.view.default(div_28, [768, 128, 128]);  div_28 = None
        bmm_26: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_157, view_229);  permute_157 = None
        bmm_27: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_229, permute_158);  view_229 = permute_158 = None
        view_230: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_26, [64, 12, 64, 128]);  bmm_26 = None
        view_231: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_27, [64, 12, 128, 64]);  bmm_27 = None
        permute_159: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_230, [0, 1, 3, 2]);  view_230 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_160: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_49: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        _unsafe_view_61: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_49, [64, 128, 768]);  clone_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_161: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_50: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        _unsafe_view_62: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_50, [64, 128, 768]);  clone_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_232: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_62, [8192, 768]);  _unsafe_view_62 = None
        mm_10: f32[8192, 768] = torch.ops.aten.mm.default(view_232, permute_162);  permute_162 = None
        permute_163: f32[768, 8192] = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_11: f32[768, 768] = torch.ops.aten.mm.default(permute_163, view_187);  permute_163 = None
        permute_164: f32[768, 768] = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        sum_33: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
        view_233: f32[768] = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
        view_234: f32[64, 128, 768] = torch.ops.aten.view.default(mm_10, [64, 128, 768]);  mm_10 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_184: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_353, view_234);  mul_353 = view_234 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_165: f32[768, 768] = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_166: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_235: f32[64, 128, 768] = torch.ops.aten.view.default(permute_166, [64, 128, 768]);  permute_166 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_51: f32[64, 128, 768] = torch.ops.aten.clone.default(view_235, memory_format = torch.contiguous_format);  view_235 = None
        _unsafe_view_63: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_51, [8192, 768]);  clone_51 = None
        mm_12: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_63, permute_167);  permute_167 = None
        permute_168: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_63, [1, 0])
        mm_13: f32[768, 768] = torch.ops.aten.mm.default(permute_168, view_187);  permute_168 = None
        permute_169: f32[768, 768] = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_34: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_63, [0], True);  _unsafe_view_63 = None
        view_236: f32[768] = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
        view_237: f32[64, 128, 768] = torch.ops.aten.view.default(mm_12, [64, 128, 768]);  mm_12 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_185: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_184, view_237);  add_184 = view_237 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_170: f32[768, 768] = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_238: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_61, [8192, 768]);  _unsafe_view_61 = None
        mm_14: f32[8192, 768] = torch.ops.aten.mm.default(view_238, permute_171);  permute_171 = None
        permute_172: f32[768, 8192] = torch.ops.aten.permute.default(view_238, [1, 0])
        mm_15: f32[768, 768] = torch.ops.aten.mm.default(permute_172, view_187);  permute_172 = view_187 = None
        permute_173: f32[768, 768] = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
        sum_35: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_238, [0], True);  view_238 = None
        view_239: f32[768] = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
        view_240: f32[64, 128, 768] = torch.ops.aten.view.default(mm_14, [64, 128, 768]);  mm_14 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_186: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_185, view_240);  add_185 = view_240 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_174: f32[768, 768] = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_362: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_186, primals_180);  primals_180 = None
        mul_363: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_362, 768)
        sum_36: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_362, [2], True)
        mul_364: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_362, mul_256);  mul_362 = None
        sum_37: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_364, [2], True);  mul_364 = None
        mul_365: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_256, sum_37);  sum_37 = None
        sub_68: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_363, sum_36);  mul_363 = sum_36 = None
        sub_69: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_68, mul_365);  sub_68 = mul_365 = None
        mul_366: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_29, sub_69);  div_29 = sub_69 = None
        mul_367: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_186, mul_256);  mul_256 = None
        sum_38: f32[768] = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1]);  mul_367 = None
        sum_39: f32[768] = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_3: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_33, dtype = torch.float32);  gt_33 = None
        mul_368: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_3, 1.1111111111111112);  _to_copy_3 = None
        mul_369: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_366, mul_368);  mul_368 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_241: f32[8192, 768] = torch.ops.aten.view.default(mul_369, [8192, 768]);  mul_369 = None
        mm_16: f32[8192, 3072] = torch.ops.aten.mm.default(view_241, permute_175);  permute_175 = None
        permute_176: f32[768, 8192] = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_17: f32[768, 3072] = torch.ops.aten.mm.default(permute_176, view_185);  permute_176 = view_185 = None
        permute_177: f32[3072, 768] = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
        sum_40: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
        view_242: f32[768] = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
        view_243: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_16, [64, 128, 3072]);  mm_16 = None
        permute_178: f32[768, 3072] = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_386: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_243, add_193);  view_243 = add_193 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_244: f32[8192, 3072] = torch.ops.aten.view.default(mul_386, [8192, 3072]);  mul_386 = None
        mm_18: f32[8192, 768] = torch.ops.aten.mm.default(view_244, permute_179);  permute_179 = None
        permute_180: f32[3072, 8192] = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_19: f32[3072, 768] = torch.ops.aten.mm.default(permute_180, view_183);  permute_180 = view_183 = None
        permute_181: f32[768, 3072] = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
        sum_41: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        view_245: f32[3072] = torch.ops.aten.view.default(sum_41, [3072]);  sum_41 = None
        view_246: f32[64, 128, 768] = torch.ops.aten.view.default(mm_18, [64, 128, 768]);  mm_18 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_194: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_366, view_246);  mul_366 = view_246 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_182: f32[3072, 768] = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_388: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_194, primals_174);  primals_174 = None
        mul_389: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_388, 768)
        sum_42: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
        mul_390: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_388, mul_239);  mul_388 = None
        sum_43: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
        mul_391: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_239, sum_43);  sum_43 = None
        sub_72: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_389, sum_42);  mul_389 = sum_42 = None
        sub_73: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_72, mul_391);  sub_72 = mul_391 = None
        mul_392: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_30, sub_73);  div_30 = sub_73 = None
        mul_393: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_194, mul_239);  mul_239 = None
        sum_44: f32[768] = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
        sum_45: f32[768] = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_4: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_32, dtype = torch.float32);  gt_32 = None
        mul_394: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_4, 1.1111111111111112);  _to_copy_4 = None
        mul_395: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_392, mul_394);  mul_394 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_247: f32[8192, 768] = torch.ops.aten.view.default(mul_395, [8192, 768]);  mul_395 = None
        mm_20: f32[8192, 768] = torch.ops.aten.mm.default(view_247, permute_183);  permute_183 = None
        permute_184: f32[768, 8192] = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_21: f32[768, 768] = torch.ops.aten.mm.default(permute_184, view_181);  permute_184 = view_181 = None
        permute_185: f32[768, 768] = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
        sum_46: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
        view_248: f32[768] = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
        view_249: f32[64, 128, 768] = torch.ops.aten.view.default(mm_20, [64, 128, 768]);  mm_20 = None
        permute_186: f32[768, 768] = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_250: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_249, [64, 128, 12, 64]);  view_249 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_187: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_52: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        _unsafe_view_64: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_52, [768, 128, 64]);  clone_52 = None
        bmm_28: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_188, _unsafe_view_64);  permute_188 = None
        bmm_29: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_64, permute_189);  _unsafe_view_64 = permute_189 = None
        view_251: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_28, [64, 12, 128, 64]);  bmm_28 = None
        view_252: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_29, [64, 12, 128, 128]);  bmm_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_5: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_31, dtype = torch.float32);  gt_31 = None
        mul_396: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_5, 1.1111111111111112);  _to_copy_5 = None
        mul_397: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_252, mul_396);  view_252 = mul_396 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_398: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_397, alias_85);  mul_397 = None
        sum_47: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_398, [-1], True)
        mul_399: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_85, sum_47);  alias_85 = sum_47 = None
        sub_74: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_31: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_74, 8.0);  sub_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_253: f32[768, 128, 128] = torch.ops.aten.view.default(div_31, [768, 128, 128]);  div_31 = None
        bmm_30: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_190, view_253);  permute_190 = None
        bmm_31: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_253, permute_191);  view_253 = permute_191 = None
        view_254: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_30, [64, 12, 64, 128]);  bmm_30 = None
        view_255: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_31, [64, 12, 128, 64]);  bmm_31 = None
        permute_192: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_254, [0, 1, 3, 2]);  view_254 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_193: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_53: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        _unsafe_view_65: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_53, [64, 128, 768]);  clone_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_194: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_54: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        _unsafe_view_66: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_54, [64, 128, 768]);  clone_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_256: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_66, [8192, 768]);  _unsafe_view_66 = None
        mm_22: f32[8192, 768] = torch.ops.aten.mm.default(view_256, permute_195);  permute_195 = None
        permute_196: f32[768, 8192] = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_23: f32[768, 768] = torch.ops.aten.mm.default(permute_196, view_170);  permute_196 = None
        permute_197: f32[768, 768] = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
        sum_48: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        view_257: f32[768] = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
        view_258: f32[64, 128, 768] = torch.ops.aten.view.default(mm_22, [64, 128, 768]);  mm_22 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_195: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_392, view_258);  mul_392 = view_258 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_198: f32[768, 768] = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_199: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_259: f32[64, 128, 768] = torch.ops.aten.view.default(permute_199, [64, 128, 768]);  permute_199 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_55: f32[64, 128, 768] = torch.ops.aten.clone.default(view_259, memory_format = torch.contiguous_format);  view_259 = None
        _unsafe_view_67: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_55, [8192, 768]);  clone_55 = None
        mm_24: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_67, permute_200);  permute_200 = None
        permute_201: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_67, [1, 0])
        mm_25: f32[768, 768] = torch.ops.aten.mm.default(permute_201, view_170);  permute_201 = None
        permute_202: f32[768, 768] = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
        sum_49: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_67, [0], True);  _unsafe_view_67 = None
        view_260: f32[768] = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
        view_261: f32[64, 128, 768] = torch.ops.aten.view.default(mm_24, [64, 128, 768]);  mm_24 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_196: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_195, view_261);  add_195 = view_261 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_203: f32[768, 768] = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_262: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_65, [8192, 768]);  _unsafe_view_65 = None
        mm_26: f32[8192, 768] = torch.ops.aten.mm.default(view_262, permute_204);  permute_204 = None
        permute_205: f32[768, 8192] = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_27: f32[768, 768] = torch.ops.aten.mm.default(permute_205, view_170);  permute_205 = view_170 = None
        permute_206: f32[768, 768] = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
        sum_50: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
        view_263: f32[768] = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
        view_264: f32[64, 128, 768] = torch.ops.aten.view.default(mm_26, [64, 128, 768]);  mm_26 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_197: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_196, view_264);  add_196 = view_264 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_207: f32[768, 768] = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_401: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_197, primals_164);  primals_164 = None
        mul_402: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_401, 768)
        sum_51: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_401, [2], True)
        mul_403: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_401, mul_233);  mul_401 = None
        sum_52: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_403, [2], True);  mul_403 = None
        mul_404: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_233, sum_52);  sum_52 = None
        sub_76: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_402, sum_51);  mul_402 = sum_51 = None
        sub_77: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_76, mul_404);  sub_76 = mul_404 = None
        mul_405: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_32, sub_77);  div_32 = sub_77 = None
        mul_406: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_197, mul_233);  mul_233 = None
        sum_53: f32[768] = torch.ops.aten.sum.dim_IntList(mul_406, [0, 1]);  mul_406 = None
        sum_54: f32[768] = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_6: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_30, dtype = torch.float32);  gt_30 = None
        mul_407: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_6, 1.1111111111111112);  _to_copy_6 = None
        mul_408: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_405, mul_407);  mul_407 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_265: f32[8192, 768] = torch.ops.aten.view.default(mul_408, [8192, 768]);  mul_408 = None
        mm_28: f32[8192, 3072] = torch.ops.aten.mm.default(view_265, permute_208);  permute_208 = None
        permute_209: f32[768, 8192] = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_29: f32[768, 3072] = torch.ops.aten.mm.default(permute_209, view_168);  permute_209 = view_168 = None
        permute_210: f32[3072, 768] = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
        sum_55: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        view_266: f32[768] = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
        view_267: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_28, [64, 128, 3072]);  mm_28 = None
        permute_211: f32[768, 3072] = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_425: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_267, add_204);  view_267 = add_204 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_268: f32[8192, 3072] = torch.ops.aten.view.default(mul_425, [8192, 3072]);  mul_425 = None
        mm_30: f32[8192, 768] = torch.ops.aten.mm.default(view_268, permute_212);  permute_212 = None
        permute_213: f32[3072, 8192] = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_31: f32[3072, 768] = torch.ops.aten.mm.default(permute_213, view_166);  permute_213 = view_166 = None
        permute_214: f32[768, 3072] = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
        sum_56: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
        view_269: f32[3072] = torch.ops.aten.view.default(sum_56, [3072]);  sum_56 = None
        view_270: f32[64, 128, 768] = torch.ops.aten.view.default(mm_30, [64, 128, 768]);  mm_30 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_205: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_405, view_270);  mul_405 = view_270 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_215: f32[3072, 768] = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_427: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_205, primals_158);  primals_158 = None
        mul_428: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_427, 768)
        sum_57: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
        mul_429: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_427, mul_216);  mul_427 = None
        sum_58: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
        mul_430: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_216, sum_58);  sum_58 = None
        sub_80: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_428, sum_57);  mul_428 = sum_57 = None
        sub_81: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_80, mul_430);  sub_80 = mul_430 = None
        mul_431: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_33, sub_81);  div_33 = sub_81 = None
        mul_432: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_205, mul_216);  mul_216 = None
        sum_59: f32[768] = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        sum_60: f32[768] = torch.ops.aten.sum.dim_IntList(add_205, [0, 1]);  add_205 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_7: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_29, dtype = torch.float32);  gt_29 = None
        mul_433: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_7, 1.1111111111111112);  _to_copy_7 = None
        mul_434: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_271: f32[8192, 768] = torch.ops.aten.view.default(mul_434, [8192, 768]);  mul_434 = None
        mm_32: f32[8192, 768] = torch.ops.aten.mm.default(view_271, permute_216);  permute_216 = None
        permute_217: f32[768, 8192] = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_33: f32[768, 768] = torch.ops.aten.mm.default(permute_217, view_164);  permute_217 = view_164 = None
        permute_218: f32[768, 768] = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
        sum_61: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272: f32[768] = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
        view_273: f32[64, 128, 768] = torch.ops.aten.view.default(mm_32, [64, 128, 768]);  mm_32 = None
        permute_219: f32[768, 768] = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_274: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_273, [64, 128, 12, 64]);  view_273 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_220: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_56: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        _unsafe_view_68: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_56, [768, 128, 64]);  clone_56 = None
        bmm_32: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_221, _unsafe_view_68);  permute_221 = None
        bmm_33: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_68, permute_222);  _unsafe_view_68 = permute_222 = None
        view_275: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_32, [64, 12, 128, 64]);  bmm_32 = None
        view_276: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_33, [64, 12, 128, 128]);  bmm_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_8: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_28, dtype = torch.float32);  gt_28 = None
        mul_435: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_8, 1.1111111111111112);  _to_copy_8 = None
        mul_436: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_276, mul_435);  view_276 = mul_435 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_437: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_436, alias_87);  mul_436 = None
        sum_62: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_437, [-1], True)
        mul_438: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_87, sum_62);  alias_87 = sum_62 = None
        sub_82: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_34: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_82, 8.0);  sub_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_277: f32[768, 128, 128] = torch.ops.aten.view.default(div_34, [768, 128, 128]);  div_34 = None
        bmm_34: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_223, view_277);  permute_223 = None
        bmm_35: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_277, permute_224);  view_277 = permute_224 = None
        view_278: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_34, [64, 12, 64, 128]);  bmm_34 = None
        view_279: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_35, [64, 12, 128, 64]);  bmm_35 = None
        permute_225: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2]);  view_278 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_226: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_57: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        _unsafe_view_69: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_57, [64, 128, 768]);  clone_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_227: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_58: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        _unsafe_view_70: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_58, [64, 128, 768]);  clone_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_280: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_70, [8192, 768]);  _unsafe_view_70 = None
        mm_34: f32[8192, 768] = torch.ops.aten.mm.default(view_280, permute_228);  permute_228 = None
        permute_229: f32[768, 8192] = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_35: f32[768, 768] = torch.ops.aten.mm.default(permute_229, view_153);  permute_229 = None
        permute_230: f32[768, 768] = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
        sum_63: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        view_281: f32[768] = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
        view_282: f32[64, 128, 768] = torch.ops.aten.view.default(mm_34, [64, 128, 768]);  mm_34 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_206: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_431, view_282);  mul_431 = view_282 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_231: f32[768, 768] = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_232: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_283: f32[64, 128, 768] = torch.ops.aten.view.default(permute_232, [64, 128, 768]);  permute_232 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_59: f32[64, 128, 768] = torch.ops.aten.clone.default(view_283, memory_format = torch.contiguous_format);  view_283 = None
        _unsafe_view_71: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_59, [8192, 768]);  clone_59 = None
        mm_36: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_71, permute_233);  permute_233 = None
        permute_234: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_71, [1, 0])
        mm_37: f32[768, 768] = torch.ops.aten.mm.default(permute_234, view_153);  permute_234 = None
        permute_235: f32[768, 768] = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
        sum_64: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_71, [0], True);  _unsafe_view_71 = None
        view_284: f32[768] = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
        view_285: f32[64, 128, 768] = torch.ops.aten.view.default(mm_36, [64, 128, 768]);  mm_36 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_207: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_206, view_285);  add_206 = view_285 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_236: f32[768, 768] = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_286: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_69, [8192, 768]);  _unsafe_view_69 = None
        mm_38: f32[8192, 768] = torch.ops.aten.mm.default(view_286, permute_237);  permute_237 = None
        permute_238: f32[768, 8192] = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_39: f32[768, 768] = torch.ops.aten.mm.default(permute_238, view_153);  permute_238 = view_153 = None
        permute_239: f32[768, 768] = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
        sum_65: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_286, [0], True);  view_286 = None
        view_287: f32[768] = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
        view_288: f32[64, 128, 768] = torch.ops.aten.view.default(mm_38, [64, 128, 768]);  mm_38 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_208: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_207, view_288);  add_207 = view_288 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_240: f32[768, 768] = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_440: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_208, primals_148);  primals_148 = None
        mul_441: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_440, 768)
        sum_66: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
        mul_442: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_440, mul_210);  mul_440 = None
        sum_67: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
        mul_443: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_210, sum_67);  sum_67 = None
        sub_84: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_441, sum_66);  mul_441 = sum_66 = None
        sub_85: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_84, mul_443);  sub_84 = mul_443 = None
        mul_444: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_35, sub_85);  div_35 = sub_85 = None
        mul_445: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_208, mul_210);  mul_210 = None
        sum_68: f32[768] = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
        sum_69: f32[768] = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_9: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_27, dtype = torch.float32);  gt_27 = None
        mul_446: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_9, 1.1111111111111112);  _to_copy_9 = None
        mul_447: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_444, mul_446);  mul_446 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_289: f32[8192, 768] = torch.ops.aten.view.default(mul_447, [8192, 768]);  mul_447 = None
        mm_40: f32[8192, 3072] = torch.ops.aten.mm.default(view_289, permute_241);  permute_241 = None
        permute_242: f32[768, 8192] = torch.ops.aten.permute.default(view_289, [1, 0])
        mm_41: f32[768, 3072] = torch.ops.aten.mm.default(permute_242, view_151);  permute_242 = view_151 = None
        permute_243: f32[3072, 768] = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
        sum_70: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
        view_290: f32[768] = torch.ops.aten.view.default(sum_70, [768]);  sum_70 = None
        view_291: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_40, [64, 128, 3072]);  mm_40 = None
        permute_244: f32[768, 3072] = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_464: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_291, add_215);  view_291 = add_215 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_292: f32[8192, 3072] = torch.ops.aten.view.default(mul_464, [8192, 3072]);  mul_464 = None
        mm_42: f32[8192, 768] = torch.ops.aten.mm.default(view_292, permute_245);  permute_245 = None
        permute_246: f32[3072, 8192] = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_43: f32[3072, 768] = torch.ops.aten.mm.default(permute_246, view_149);  permute_246 = view_149 = None
        permute_247: f32[768, 3072] = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
        sum_71: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
        view_293: f32[3072] = torch.ops.aten.view.default(sum_71, [3072]);  sum_71 = None
        view_294: f32[64, 128, 768] = torch.ops.aten.view.default(mm_42, [64, 128, 768]);  mm_42 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_216: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_444, view_294);  mul_444 = view_294 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_248: f32[3072, 768] = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_466: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_216, primals_142);  primals_142 = None
        mul_467: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_466, 768)
        sum_72: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_466, [2], True)
        mul_468: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_466, mul_193);  mul_466 = None
        sum_73: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_468, [2], True);  mul_468 = None
        mul_469: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_193, sum_73);  sum_73 = None
        sub_88: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_467, sum_72);  mul_467 = sum_72 = None
        sub_89: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_88, mul_469);  sub_88 = mul_469 = None
        mul_470: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_36, sub_89);  div_36 = sub_89 = None
        mul_471: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_216, mul_193);  mul_193 = None
        sum_74: f32[768] = torch.ops.aten.sum.dim_IntList(mul_471, [0, 1]);  mul_471 = None
        sum_75: f32[768] = torch.ops.aten.sum.dim_IntList(add_216, [0, 1]);  add_216 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_10: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_26, dtype = torch.float32);  gt_26 = None
        mul_472: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_10, 1.1111111111111112);  _to_copy_10 = None
        mul_473: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_470, mul_472);  mul_472 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_295: f32[8192, 768] = torch.ops.aten.view.default(mul_473, [8192, 768]);  mul_473 = None
        mm_44: f32[8192, 768] = torch.ops.aten.mm.default(view_295, permute_249);  permute_249 = None
        permute_250: f32[768, 8192] = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_45: f32[768, 768] = torch.ops.aten.mm.default(permute_250, view_147);  permute_250 = view_147 = None
        permute_251: f32[768, 768] = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
        sum_76: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        view_296: f32[768] = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
        view_297: f32[64, 128, 768] = torch.ops.aten.view.default(mm_44, [64, 128, 768]);  mm_44 = None
        permute_252: f32[768, 768] = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_298: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_297, [64, 128, 12, 64]);  view_297 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_253: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_60: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        _unsafe_view_72: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_60, [768, 128, 64]);  clone_60 = None
        bmm_36: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_254, _unsafe_view_72);  permute_254 = None
        bmm_37: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_72, permute_255);  _unsafe_view_72 = permute_255 = None
        view_299: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_36, [64, 12, 128, 64]);  bmm_36 = None
        view_300: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_37, [64, 12, 128, 128]);  bmm_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_11: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_25, dtype = torch.float32);  gt_25 = None
        mul_474: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_11, 1.1111111111111112);  _to_copy_11 = None
        mul_475: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_300, mul_474);  view_300 = mul_474 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_476: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_475, alias_89);  mul_475 = None
        sum_77: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
        mul_477: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_89, sum_77);  alias_89 = sum_77 = None
        sub_90: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_37: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_90, 8.0);  sub_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_301: f32[768, 128, 128] = torch.ops.aten.view.default(div_37, [768, 128, 128]);  div_37 = None
        bmm_38: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_256, view_301);  permute_256 = None
        bmm_39: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_301, permute_257);  view_301 = permute_257 = None
        view_302: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_38, [64, 12, 64, 128]);  bmm_38 = None
        view_303: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_39, [64, 12, 128, 64]);  bmm_39 = None
        permute_258: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_302, [0, 1, 3, 2]);  view_302 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_259: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_61: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        _unsafe_view_73: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_61, [64, 128, 768]);  clone_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_260: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_62: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        _unsafe_view_74: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_62, [64, 128, 768]);  clone_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_304: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_74, [8192, 768]);  _unsafe_view_74 = None
        mm_46: f32[8192, 768] = torch.ops.aten.mm.default(view_304, permute_261);  permute_261 = None
        permute_262: f32[768, 8192] = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_47: f32[768, 768] = torch.ops.aten.mm.default(permute_262, view_136);  permute_262 = None
        permute_263: f32[768, 768] = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
        sum_78: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        view_305: f32[768] = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
        view_306: f32[64, 128, 768] = torch.ops.aten.view.default(mm_46, [64, 128, 768]);  mm_46 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_217: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_470, view_306);  mul_470 = view_306 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_264: f32[768, 768] = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_265: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_307: f32[64, 128, 768] = torch.ops.aten.view.default(permute_265, [64, 128, 768]);  permute_265 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_63: f32[64, 128, 768] = torch.ops.aten.clone.default(view_307, memory_format = torch.contiguous_format);  view_307 = None
        _unsafe_view_75: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_63, [8192, 768]);  clone_63 = None
        mm_48: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_75, permute_266);  permute_266 = None
        permute_267: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_75, [1, 0])
        mm_49: f32[768, 768] = torch.ops.aten.mm.default(permute_267, view_136);  permute_267 = None
        permute_268: f32[768, 768] = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
        sum_79: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_75, [0], True);  _unsafe_view_75 = None
        view_308: f32[768] = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
        view_309: f32[64, 128, 768] = torch.ops.aten.view.default(mm_48, [64, 128, 768]);  mm_48 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_218: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_217, view_309);  add_217 = view_309 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_269: f32[768, 768] = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_310: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_73, [8192, 768]);  _unsafe_view_73 = None
        mm_50: f32[8192, 768] = torch.ops.aten.mm.default(view_310, permute_270);  permute_270 = None
        permute_271: f32[768, 8192] = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_51: f32[768, 768] = torch.ops.aten.mm.default(permute_271, view_136);  permute_271 = view_136 = None
        permute_272: f32[768, 768] = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
        sum_80: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        view_311: f32[768] = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
        view_312: f32[64, 128, 768] = torch.ops.aten.view.default(mm_50, [64, 128, 768]);  mm_50 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_219: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_218, view_312);  add_218 = view_312 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_273: f32[768, 768] = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_479: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_219, primals_132);  primals_132 = None
        mul_480: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_479, 768)
        sum_81: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_479, [2], True)
        mul_481: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_479, mul_187);  mul_479 = None
        sum_82: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_481, [2], True);  mul_481 = None
        mul_482: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_187, sum_82);  sum_82 = None
        sub_92: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_480, sum_81);  mul_480 = sum_81 = None
        sub_93: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_92, mul_482);  sub_92 = mul_482 = None
        mul_483: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_38, sub_93);  div_38 = sub_93 = None
        mul_484: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_219, mul_187);  mul_187 = None
        sum_83: f32[768] = torch.ops.aten.sum.dim_IntList(mul_484, [0, 1]);  mul_484 = None
        sum_84: f32[768] = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_12: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_24, dtype = torch.float32);  gt_24 = None
        mul_485: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_12, 1.1111111111111112);  _to_copy_12 = None
        mul_486: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_483, mul_485);  mul_485 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_313: f32[8192, 768] = torch.ops.aten.view.default(mul_486, [8192, 768]);  mul_486 = None
        mm_52: f32[8192, 3072] = torch.ops.aten.mm.default(view_313, permute_274);  permute_274 = None
        permute_275: f32[768, 8192] = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_53: f32[768, 3072] = torch.ops.aten.mm.default(permute_275, view_134);  permute_275 = view_134 = None
        permute_276: f32[3072, 768] = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
        sum_85: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        view_314: f32[768] = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
        view_315: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_52, [64, 128, 3072]);  mm_52 = None
        permute_277: f32[768, 3072] = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_503: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_315, add_226);  view_315 = add_226 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_316: f32[8192, 3072] = torch.ops.aten.view.default(mul_503, [8192, 3072]);  mul_503 = None
        mm_54: f32[8192, 768] = torch.ops.aten.mm.default(view_316, permute_278);  permute_278 = None
        permute_279: f32[3072, 8192] = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_55: f32[3072, 768] = torch.ops.aten.mm.default(permute_279, view_132);  permute_279 = view_132 = None
        permute_280: f32[768, 3072] = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
        sum_86: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        view_317: f32[3072] = torch.ops.aten.view.default(sum_86, [3072]);  sum_86 = None
        view_318: f32[64, 128, 768] = torch.ops.aten.view.default(mm_54, [64, 128, 768]);  mm_54 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_227: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_483, view_318);  mul_483 = view_318 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_281: f32[3072, 768] = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_505: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_227, primals_126);  primals_126 = None
        mul_506: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_505, 768)
        sum_87: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_505, [2], True)
        mul_507: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_505, mul_170);  mul_505 = None
        sum_88: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_507, [2], True);  mul_507 = None
        mul_508: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_170, sum_88);  sum_88 = None
        sub_96: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_506, sum_87);  mul_506 = sum_87 = None
        sub_97: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_96, mul_508);  sub_96 = mul_508 = None
        mul_509: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_39, sub_97);  div_39 = sub_97 = None
        mul_510: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_227, mul_170);  mul_170 = None
        sum_89: f32[768] = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1]);  mul_510 = None
        sum_90: f32[768] = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_13: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_23, dtype = torch.float32);  gt_23 = None
        mul_511: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_13, 1.1111111111111112);  _to_copy_13 = None
        mul_512: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_509, mul_511);  mul_511 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_319: f32[8192, 768] = torch.ops.aten.view.default(mul_512, [8192, 768]);  mul_512 = None
        mm_56: f32[8192, 768] = torch.ops.aten.mm.default(view_319, permute_282);  permute_282 = None
        permute_283: f32[768, 8192] = torch.ops.aten.permute.default(view_319, [1, 0])
        mm_57: f32[768, 768] = torch.ops.aten.mm.default(permute_283, view_130);  permute_283 = view_130 = None
        permute_284: f32[768, 768] = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
        sum_91: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
        view_320: f32[768] = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
        view_321: f32[64, 128, 768] = torch.ops.aten.view.default(mm_56, [64, 128, 768]);  mm_56 = None
        permute_285: f32[768, 768] = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_322: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_321, [64, 128, 12, 64]);  view_321 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_286: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_64: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        _unsafe_view_76: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_64, [768, 128, 64]);  clone_64 = None
        bmm_40: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_287, _unsafe_view_76);  permute_287 = None
        bmm_41: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_76, permute_288);  _unsafe_view_76 = permute_288 = None
        view_323: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_40, [64, 12, 128, 64]);  bmm_40 = None
        view_324: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_41, [64, 12, 128, 128]);  bmm_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_14: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_22, dtype = torch.float32);  gt_22 = None
        mul_513: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_14, 1.1111111111111112);  _to_copy_14 = None
        mul_514: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_324, mul_513);  view_324 = mul_513 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_515: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_514, alias_91);  mul_514 = None
        sum_92: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_515, [-1], True)
        mul_516: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_91, sum_92);  alias_91 = sum_92 = None
        sub_98: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_40: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_98, 8.0);  sub_98 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_325: f32[768, 128, 128] = torch.ops.aten.view.default(div_40, [768, 128, 128]);  div_40 = None
        bmm_42: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_289, view_325);  permute_289 = None
        bmm_43: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_325, permute_290);  view_325 = permute_290 = None
        view_326: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_42, [64, 12, 64, 128]);  bmm_42 = None
        view_327: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_43, [64, 12, 128, 64]);  bmm_43 = None
        permute_291: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_326, [0, 1, 3, 2]);  view_326 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_292: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_65: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        _unsafe_view_77: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_65, [64, 128, 768]);  clone_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_293: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_66: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        _unsafe_view_78: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_66, [64, 128, 768]);  clone_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_328: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_78, [8192, 768]);  _unsafe_view_78 = None
        mm_58: f32[8192, 768] = torch.ops.aten.mm.default(view_328, permute_294);  permute_294 = None
        permute_295: f32[768, 8192] = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_59: f32[768, 768] = torch.ops.aten.mm.default(permute_295, view_119);  permute_295 = None
        permute_296: f32[768, 768] = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
        sum_93: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        view_329: f32[768] = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
        view_330: f32[64, 128, 768] = torch.ops.aten.view.default(mm_58, [64, 128, 768]);  mm_58 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_228: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_509, view_330);  mul_509 = view_330 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_297: f32[768, 768] = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_298: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_331: f32[64, 128, 768] = torch.ops.aten.view.default(permute_298, [64, 128, 768]);  permute_298 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_67: f32[64, 128, 768] = torch.ops.aten.clone.default(view_331, memory_format = torch.contiguous_format);  view_331 = None
        _unsafe_view_79: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_67, [8192, 768]);  clone_67 = None
        mm_60: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_79, permute_299);  permute_299 = None
        permute_300: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_79, [1, 0])
        mm_61: f32[768, 768] = torch.ops.aten.mm.default(permute_300, view_119);  permute_300 = None
        permute_301: f32[768, 768] = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
        sum_94: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_79, [0], True);  _unsafe_view_79 = None
        view_332: f32[768] = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
        view_333: f32[64, 128, 768] = torch.ops.aten.view.default(mm_60, [64, 128, 768]);  mm_60 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_229: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_228, view_333);  add_228 = view_333 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_302: f32[768, 768] = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_334: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_77, [8192, 768]);  _unsafe_view_77 = None
        mm_62: f32[8192, 768] = torch.ops.aten.mm.default(view_334, permute_303);  permute_303 = None
        permute_304: f32[768, 8192] = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_63: f32[768, 768] = torch.ops.aten.mm.default(permute_304, view_119);  permute_304 = view_119 = None
        permute_305: f32[768, 768] = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
        sum_95: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        view_335: f32[768] = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
        view_336: f32[64, 128, 768] = torch.ops.aten.view.default(mm_62, [64, 128, 768]);  mm_62 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_230: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_229, view_336);  add_229 = view_336 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_306: f32[768, 768] = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_518: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_230, primals_116);  primals_116 = None
        mul_519: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_518, 768)
        sum_96: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_518, [2], True)
        mul_520: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_518, mul_164);  mul_518 = None
        sum_97: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_520, [2], True);  mul_520 = None
        mul_521: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_164, sum_97);  sum_97 = None
        sub_100: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_519, sum_96);  mul_519 = sum_96 = None
        sub_101: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_100, mul_521);  sub_100 = mul_521 = None
        mul_522: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_41, sub_101);  div_41 = sub_101 = None
        mul_523: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_230, mul_164);  mul_164 = None
        sum_98: f32[768] = torch.ops.aten.sum.dim_IntList(mul_523, [0, 1]);  mul_523 = None
        sum_99: f32[768] = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_15: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_21, dtype = torch.float32);  gt_21 = None
        mul_524: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_15, 1.1111111111111112);  _to_copy_15 = None
        mul_525: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_522, mul_524);  mul_524 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_337: f32[8192, 768] = torch.ops.aten.view.default(mul_525, [8192, 768]);  mul_525 = None
        mm_64: f32[8192, 3072] = torch.ops.aten.mm.default(view_337, permute_307);  permute_307 = None
        permute_308: f32[768, 8192] = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_65: f32[768, 3072] = torch.ops.aten.mm.default(permute_308, view_117);  permute_308 = view_117 = None
        permute_309: f32[3072, 768] = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
        sum_100: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        view_338: f32[768] = torch.ops.aten.view.default(sum_100, [768]);  sum_100 = None
        view_339: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_64, [64, 128, 3072]);  mm_64 = None
        permute_310: f32[768, 3072] = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_542: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_339, add_237);  view_339 = add_237 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_340: f32[8192, 3072] = torch.ops.aten.view.default(mul_542, [8192, 3072]);  mul_542 = None
        mm_66: f32[8192, 768] = torch.ops.aten.mm.default(view_340, permute_311);  permute_311 = None
        permute_312: f32[3072, 8192] = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_67: f32[3072, 768] = torch.ops.aten.mm.default(permute_312, view_115);  permute_312 = view_115 = None
        permute_313: f32[768, 3072] = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
        sum_101: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
        view_341: f32[3072] = torch.ops.aten.view.default(sum_101, [3072]);  sum_101 = None
        view_342: f32[64, 128, 768] = torch.ops.aten.view.default(mm_66, [64, 128, 768]);  mm_66 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_238: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_522, view_342);  mul_522 = view_342 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_314: f32[3072, 768] = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_544: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_238, primals_110);  primals_110 = None
        mul_545: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_544, 768)
        sum_102: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_544, [2], True)
        mul_546: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_544, mul_147);  mul_544 = None
        sum_103: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_546, [2], True);  mul_546 = None
        mul_547: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_147, sum_103);  sum_103 = None
        sub_104: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_545, sum_102);  mul_545 = sum_102 = None
        sub_105: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_104, mul_547);  sub_104 = mul_547 = None
        mul_548: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_42, sub_105);  div_42 = sub_105 = None
        mul_549: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_238, mul_147);  mul_147 = None
        sum_104: f32[768] = torch.ops.aten.sum.dim_IntList(mul_549, [0, 1]);  mul_549 = None
        sum_105: f32[768] = torch.ops.aten.sum.dim_IntList(add_238, [0, 1]);  add_238 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_16: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_20, dtype = torch.float32);  gt_20 = None
        mul_550: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_16, 1.1111111111111112);  _to_copy_16 = None
        mul_551: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_548, mul_550);  mul_550 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_343: f32[8192, 768] = torch.ops.aten.view.default(mul_551, [8192, 768]);  mul_551 = None
        mm_68: f32[8192, 768] = torch.ops.aten.mm.default(view_343, permute_315);  permute_315 = None
        permute_316: f32[768, 8192] = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_69: f32[768, 768] = torch.ops.aten.mm.default(permute_316, view_113);  permute_316 = view_113 = None
        permute_317: f32[768, 768] = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
        sum_106: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344: f32[768] = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
        view_345: f32[64, 128, 768] = torch.ops.aten.view.default(mm_68, [64, 128, 768]);  mm_68 = None
        permute_318: f32[768, 768] = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_346: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_345, [64, 128, 12, 64]);  view_345 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_319: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_68: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        _unsafe_view_80: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_68, [768, 128, 64]);  clone_68 = None
        bmm_44: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_320, _unsafe_view_80);  permute_320 = None
        bmm_45: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_80, permute_321);  _unsafe_view_80 = permute_321 = None
        view_347: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_44, [64, 12, 128, 64]);  bmm_44 = None
        view_348: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_45, [64, 12, 128, 128]);  bmm_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_17: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_19, dtype = torch.float32);  gt_19 = None
        mul_552: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_17, 1.1111111111111112);  _to_copy_17 = None
        mul_553: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_348, mul_552);  view_348 = mul_552 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_554: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_553, alias_93);  mul_553 = None
        sum_107: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_554, [-1], True)
        mul_555: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_93, sum_107);  alias_93 = sum_107 = None
        sub_106: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_43: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_106, 8.0);  sub_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_349: f32[768, 128, 128] = torch.ops.aten.view.default(div_43, [768, 128, 128]);  div_43 = None
        bmm_46: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_322, view_349);  permute_322 = None
        bmm_47: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_349, permute_323);  view_349 = permute_323 = None
        view_350: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_46, [64, 12, 64, 128]);  bmm_46 = None
        view_351: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_47, [64, 12, 128, 64]);  bmm_47 = None
        permute_324: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2]);  view_350 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_325: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_69: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        _unsafe_view_81: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_69, [64, 128, 768]);  clone_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_326: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_70: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        _unsafe_view_82: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_70, [64, 128, 768]);  clone_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_352: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_82, [8192, 768]);  _unsafe_view_82 = None
        mm_70: f32[8192, 768] = torch.ops.aten.mm.default(view_352, permute_327);  permute_327 = None
        permute_328: f32[768, 8192] = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_71: f32[768, 768] = torch.ops.aten.mm.default(permute_328, view_102);  permute_328 = None
        permute_329: f32[768, 768] = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
        sum_108: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        view_353: f32[768] = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
        view_354: f32[64, 128, 768] = torch.ops.aten.view.default(mm_70, [64, 128, 768]);  mm_70 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_239: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_548, view_354);  mul_548 = view_354 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_330: f32[768, 768] = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_331: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_355: f32[64, 128, 768] = torch.ops.aten.view.default(permute_331, [64, 128, 768]);  permute_331 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_71: f32[64, 128, 768] = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
        _unsafe_view_83: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_71, [8192, 768]);  clone_71 = None
        mm_72: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_83, permute_332);  permute_332 = None
        permute_333: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_83, [1, 0])
        mm_73: f32[768, 768] = torch.ops.aten.mm.default(permute_333, view_102);  permute_333 = None
        permute_334: f32[768, 768] = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
        sum_109: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_83, [0], True);  _unsafe_view_83 = None
        view_356: f32[768] = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
        view_357: f32[64, 128, 768] = torch.ops.aten.view.default(mm_72, [64, 128, 768]);  mm_72 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_240: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_239, view_357);  add_239 = view_357 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_335: f32[768, 768] = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_358: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_81, [8192, 768]);  _unsafe_view_81 = None
        mm_74: f32[8192, 768] = torch.ops.aten.mm.default(view_358, permute_336);  permute_336 = None
        permute_337: f32[768, 8192] = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_75: f32[768, 768] = torch.ops.aten.mm.default(permute_337, view_102);  permute_337 = view_102 = None
        permute_338: f32[768, 768] = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
        sum_110: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
        view_359: f32[768] = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
        view_360: f32[64, 128, 768] = torch.ops.aten.view.default(mm_74, [64, 128, 768]);  mm_74 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_241: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_240, view_360);  add_240 = view_360 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_339: f32[768, 768] = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_557: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_241, primals_100);  primals_100 = None
        mul_558: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_557, 768)
        sum_111: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
        mul_559: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_557, mul_141);  mul_557 = None
        sum_112: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
        mul_560: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_141, sum_112);  sum_112 = None
        sub_108: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_558, sum_111);  mul_558 = sum_111 = None
        sub_109: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_108, mul_560);  sub_108 = mul_560 = None
        mul_561: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_44, sub_109);  div_44 = sub_109 = None
        mul_562: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_241, mul_141);  mul_141 = None
        sum_113: f32[768] = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
        sum_114: f32[768] = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_18: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_18, dtype = torch.float32);  gt_18 = None
        mul_563: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_18, 1.1111111111111112);  _to_copy_18 = None
        mul_564: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_561, mul_563);  mul_563 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_361: f32[8192, 768] = torch.ops.aten.view.default(mul_564, [8192, 768]);  mul_564 = None
        mm_76: f32[8192, 3072] = torch.ops.aten.mm.default(view_361, permute_340);  permute_340 = None
        permute_341: f32[768, 8192] = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_77: f32[768, 3072] = torch.ops.aten.mm.default(permute_341, view_100);  permute_341 = view_100 = None
        permute_342: f32[3072, 768] = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
        sum_115: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        view_362: f32[768] = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
        view_363: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_76, [64, 128, 3072]);  mm_76 = None
        permute_343: f32[768, 3072] = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_581: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_363, add_248);  view_363 = add_248 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_364: f32[8192, 3072] = torch.ops.aten.view.default(mul_581, [8192, 3072]);  mul_581 = None
        mm_78: f32[8192, 768] = torch.ops.aten.mm.default(view_364, permute_344);  permute_344 = None
        permute_345: f32[3072, 8192] = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_79: f32[3072, 768] = torch.ops.aten.mm.default(permute_345, view_98);  permute_345 = view_98 = None
        permute_346: f32[768, 3072] = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
        sum_116: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        view_365: f32[3072] = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
        view_366: f32[64, 128, 768] = torch.ops.aten.view.default(mm_78, [64, 128, 768]);  mm_78 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_249: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_561, view_366);  mul_561 = view_366 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_347: f32[3072, 768] = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_583: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_249, primals_94);  primals_94 = None
        mul_584: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_583, 768)
        sum_117: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_583, [2], True)
        mul_585: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_583, mul_124);  mul_583 = None
        sum_118: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_585, [2], True);  mul_585 = None
        mul_586: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_124, sum_118);  sum_118 = None
        sub_112: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_584, sum_117);  mul_584 = sum_117 = None
        sub_113: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_112, mul_586);  sub_112 = mul_586 = None
        mul_587: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_45, sub_113);  div_45 = sub_113 = None
        mul_588: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_249, mul_124);  mul_124 = None
        sum_119: f32[768] = torch.ops.aten.sum.dim_IntList(mul_588, [0, 1]);  mul_588 = None
        sum_120: f32[768] = torch.ops.aten.sum.dim_IntList(add_249, [0, 1]);  add_249 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_19: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_17, dtype = torch.float32);  gt_17 = None
        mul_589: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_19, 1.1111111111111112);  _to_copy_19 = None
        mul_590: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_587, mul_589);  mul_589 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_367: f32[8192, 768] = torch.ops.aten.view.default(mul_590, [8192, 768]);  mul_590 = None
        mm_80: f32[8192, 768] = torch.ops.aten.mm.default(view_367, permute_348);  permute_348 = None
        permute_349: f32[768, 8192] = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_81: f32[768, 768] = torch.ops.aten.mm.default(permute_349, view_96);  permute_349 = view_96 = None
        permute_350: f32[768, 768] = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        sum_121: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368: f32[768] = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
        view_369: f32[64, 128, 768] = torch.ops.aten.view.default(mm_80, [64, 128, 768]);  mm_80 = None
        permute_351: f32[768, 768] = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_370: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_369, [64, 128, 12, 64]);  view_369 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_352: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_72: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        _unsafe_view_84: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_72, [768, 128, 64]);  clone_72 = None
        bmm_48: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_353, _unsafe_view_84);  permute_353 = None
        bmm_49: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_84, permute_354);  _unsafe_view_84 = permute_354 = None
        view_371: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_48, [64, 12, 128, 64]);  bmm_48 = None
        view_372: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_49, [64, 12, 128, 128]);  bmm_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_20: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_16, dtype = torch.float32);  gt_16 = None
        mul_591: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_20, 1.1111111111111112);  _to_copy_20 = None
        mul_592: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_372, mul_591);  view_372 = mul_591 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_593: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_592, alias_95);  mul_592 = None
        sum_122: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_593, [-1], True)
        mul_594: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_95, sum_122);  alias_95 = sum_122 = None
        sub_114: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_46: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_114, 8.0);  sub_114 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_373: f32[768, 128, 128] = torch.ops.aten.view.default(div_46, [768, 128, 128]);  div_46 = None
        bmm_50: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_355, view_373);  permute_355 = None
        bmm_51: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_373, permute_356);  view_373 = permute_356 = None
        view_374: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_50, [64, 12, 64, 128]);  bmm_50 = None
        view_375: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_51, [64, 12, 128, 64]);  bmm_51 = None
        permute_357: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_374, [0, 1, 3, 2]);  view_374 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_358: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_73: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        _unsafe_view_85: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_73, [64, 128, 768]);  clone_73 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_359: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_74: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        _unsafe_view_86: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_74, [64, 128, 768]);  clone_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_376: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_86, [8192, 768]);  _unsafe_view_86 = None
        mm_82: f32[8192, 768] = torch.ops.aten.mm.default(view_376, permute_360);  permute_360 = None
        permute_361: f32[768, 8192] = torch.ops.aten.permute.default(view_376, [1, 0])
        mm_83: f32[768, 768] = torch.ops.aten.mm.default(permute_361, view_85);  permute_361 = None
        permute_362: f32[768, 768] = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
        sum_123: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_376, [0], True);  view_376 = None
        view_377: f32[768] = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
        view_378: f32[64, 128, 768] = torch.ops.aten.view.default(mm_82, [64, 128, 768]);  mm_82 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_250: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_587, view_378);  mul_587 = view_378 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_363: f32[768, 768] = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_364: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_379: f32[64, 128, 768] = torch.ops.aten.view.default(permute_364, [64, 128, 768]);  permute_364 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_75: f32[64, 128, 768] = torch.ops.aten.clone.default(view_379, memory_format = torch.contiguous_format);  view_379 = None
        _unsafe_view_87: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_75, [8192, 768]);  clone_75 = None
        mm_84: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_87, permute_365);  permute_365 = None
        permute_366: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_87, [1, 0])
        mm_85: f32[768, 768] = torch.ops.aten.mm.default(permute_366, view_85);  permute_366 = None
        permute_367: f32[768, 768] = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
        sum_124: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_87, [0], True);  _unsafe_view_87 = None
        view_380: f32[768] = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
        view_381: f32[64, 128, 768] = torch.ops.aten.view.default(mm_84, [64, 128, 768]);  mm_84 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_251: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_250, view_381);  add_250 = view_381 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_368: f32[768, 768] = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_382: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_85, [8192, 768]);  _unsafe_view_85 = None
        mm_86: f32[8192, 768] = torch.ops.aten.mm.default(view_382, permute_369);  permute_369 = None
        permute_370: f32[768, 8192] = torch.ops.aten.permute.default(view_382, [1, 0])
        mm_87: f32[768, 768] = torch.ops.aten.mm.default(permute_370, view_85);  permute_370 = view_85 = None
        permute_371: f32[768, 768] = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
        sum_125: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
        view_383: f32[768] = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
        view_384: f32[64, 128, 768] = torch.ops.aten.view.default(mm_86, [64, 128, 768]);  mm_86 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_252: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_251, view_384);  add_251 = view_384 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_372: f32[768, 768] = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_596: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_252, primals_84);  primals_84 = None
        mul_597: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_596, 768)
        sum_126: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_596, [2], True)
        mul_598: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_596, mul_118);  mul_596 = None
        sum_127: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        mul_599: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_118, sum_127);  sum_127 = None
        sub_116: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_597, sum_126);  mul_597 = sum_126 = None
        sub_117: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_116, mul_599);  sub_116 = mul_599 = None
        mul_600: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_47, sub_117);  div_47 = sub_117 = None
        mul_601: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_252, mul_118);  mul_118 = None
        sum_128: f32[768] = torch.ops.aten.sum.dim_IntList(mul_601, [0, 1]);  mul_601 = None
        sum_129: f32[768] = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_21: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_15, dtype = torch.float32);  gt_15 = None
        mul_602: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_21, 1.1111111111111112);  _to_copy_21 = None
        mul_603: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_600, mul_602);  mul_602 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_385: f32[8192, 768] = torch.ops.aten.view.default(mul_603, [8192, 768]);  mul_603 = None
        mm_88: f32[8192, 3072] = torch.ops.aten.mm.default(view_385, permute_373);  permute_373 = None
        permute_374: f32[768, 8192] = torch.ops.aten.permute.default(view_385, [1, 0])
        mm_89: f32[768, 3072] = torch.ops.aten.mm.default(permute_374, view_83);  permute_374 = view_83 = None
        permute_375: f32[3072, 768] = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
        sum_130: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
        view_386: f32[768] = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
        view_387: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_88, [64, 128, 3072]);  mm_88 = None
        permute_376: f32[768, 3072] = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_620: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_387, add_259);  view_387 = add_259 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_388: f32[8192, 3072] = torch.ops.aten.view.default(mul_620, [8192, 3072]);  mul_620 = None
        mm_90: f32[8192, 768] = torch.ops.aten.mm.default(view_388, permute_377);  permute_377 = None
        permute_378: f32[3072, 8192] = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_91: f32[3072, 768] = torch.ops.aten.mm.default(permute_378, view_81);  permute_378 = view_81 = None
        permute_379: f32[768, 3072] = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        sum_131: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        view_389: f32[3072] = torch.ops.aten.view.default(sum_131, [3072]);  sum_131 = None
        view_390: f32[64, 128, 768] = torch.ops.aten.view.default(mm_90, [64, 128, 768]);  mm_90 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_260: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_600, view_390);  mul_600 = view_390 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_380: f32[3072, 768] = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_622: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_260, primals_78);  primals_78 = None
        mul_623: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_622, 768)
        sum_132: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_622, [2], True)
        mul_624: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_622, mul_101);  mul_622 = None
        sum_133: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_624, [2], True);  mul_624 = None
        mul_625: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_101, sum_133);  sum_133 = None
        sub_120: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_623, sum_132);  mul_623 = sum_132 = None
        sub_121: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_120, mul_625);  sub_120 = mul_625 = None
        mul_626: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_48, sub_121);  div_48 = sub_121 = None
        mul_627: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_260, mul_101);  mul_101 = None
        sum_134: f32[768] = torch.ops.aten.sum.dim_IntList(mul_627, [0, 1]);  mul_627 = None
        sum_135: f32[768] = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_22: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_14, dtype = torch.float32);  gt_14 = None
        mul_628: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_22, 1.1111111111111112);  _to_copy_22 = None
        mul_629: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_626, mul_628);  mul_628 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_391: f32[8192, 768] = torch.ops.aten.view.default(mul_629, [8192, 768]);  mul_629 = None
        mm_92: f32[8192, 768] = torch.ops.aten.mm.default(view_391, permute_381);  permute_381 = None
        permute_382: f32[768, 8192] = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_93: f32[768, 768] = torch.ops.aten.mm.default(permute_382, view_79);  permute_382 = view_79 = None
        permute_383: f32[768, 768] = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
        sum_136: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        view_392: f32[768] = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
        view_393: f32[64, 128, 768] = torch.ops.aten.view.default(mm_92, [64, 128, 768]);  mm_92 = None
        permute_384: f32[768, 768] = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_394: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_393, [64, 128, 12, 64]);  view_393 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_385: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_76: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        _unsafe_view_88: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_76, [768, 128, 64]);  clone_76 = None
        bmm_52: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_386, _unsafe_view_88);  permute_386 = None
        bmm_53: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_88, permute_387);  _unsafe_view_88 = permute_387 = None
        view_395: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_52, [64, 12, 128, 64]);  bmm_52 = None
        view_396: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_53, [64, 12, 128, 128]);  bmm_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_23: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_13, dtype = torch.float32);  gt_13 = None
        mul_630: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_23, 1.1111111111111112);  _to_copy_23 = None
        mul_631: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_396, mul_630);  view_396 = mul_630 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_632: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_631, alias_97);  mul_631 = None
        sum_137: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_632, [-1], True)
        mul_633: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_97, sum_137);  alias_97 = sum_137 = None
        sub_122: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_49: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_397: f32[768, 128, 128] = torch.ops.aten.view.default(div_49, [768, 128, 128]);  div_49 = None
        bmm_54: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_388, view_397);  permute_388 = None
        bmm_55: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_397, permute_389);  view_397 = permute_389 = None
        view_398: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_54, [64, 12, 64, 128]);  bmm_54 = None
        view_399: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_55, [64, 12, 128, 64]);  bmm_55 = None
        permute_390: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_398, [0, 1, 3, 2]);  view_398 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_391: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_77: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        _unsafe_view_89: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_77, [64, 128, 768]);  clone_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_392: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_78: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        _unsafe_view_90: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_78, [64, 128, 768]);  clone_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_400: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_90, [8192, 768]);  _unsafe_view_90 = None
        mm_94: f32[8192, 768] = torch.ops.aten.mm.default(view_400, permute_393);  permute_393 = None
        permute_394: f32[768, 8192] = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_95: f32[768, 768] = torch.ops.aten.mm.default(permute_394, view_68);  permute_394 = None
        permute_395: f32[768, 768] = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
        sum_138: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
        view_401: f32[768] = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
        view_402: f32[64, 128, 768] = torch.ops.aten.view.default(mm_94, [64, 128, 768]);  mm_94 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_261: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_626, view_402);  mul_626 = view_402 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_396: f32[768, 768] = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_397: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_403: f32[64, 128, 768] = torch.ops.aten.view.default(permute_397, [64, 128, 768]);  permute_397 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_79: f32[64, 128, 768] = torch.ops.aten.clone.default(view_403, memory_format = torch.contiguous_format);  view_403 = None
        _unsafe_view_91: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_79, [8192, 768]);  clone_79 = None
        mm_96: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_91, permute_398);  permute_398 = None
        permute_399: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_91, [1, 0])
        mm_97: f32[768, 768] = torch.ops.aten.mm.default(permute_399, view_68);  permute_399 = None
        permute_400: f32[768, 768] = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
        sum_139: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_91, [0], True);  _unsafe_view_91 = None
        view_404: f32[768] = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
        view_405: f32[64, 128, 768] = torch.ops.aten.view.default(mm_96, [64, 128, 768]);  mm_96 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_262: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_261, view_405);  add_261 = view_405 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_401: f32[768, 768] = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_406: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_89, [8192, 768]);  _unsafe_view_89 = None
        mm_98: f32[8192, 768] = torch.ops.aten.mm.default(view_406, permute_402);  permute_402 = None
        permute_403: f32[768, 8192] = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_99: f32[768, 768] = torch.ops.aten.mm.default(permute_403, view_68);  permute_403 = view_68 = None
        permute_404: f32[768, 768] = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
        sum_140: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        view_407: f32[768] = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
        view_408: f32[64, 128, 768] = torch.ops.aten.view.default(mm_98, [64, 128, 768]);  mm_98 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_263: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_262, view_408);  add_262 = view_408 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_405: f32[768, 768] = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_635: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_263, primals_68);  primals_68 = None
        mul_636: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_635, 768)
        sum_141: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
        mul_637: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_635, mul_95);  mul_635 = None
        sum_142: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
        mul_638: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_95, sum_142);  sum_142 = None
        sub_124: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_636, sum_141);  mul_636 = sum_141 = None
        sub_125: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_124, mul_638);  sub_124 = mul_638 = None
        mul_639: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_50, sub_125);  div_50 = sub_125 = None
        mul_640: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_263, mul_95);  mul_95 = None
        sum_143: f32[768] = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
        sum_144: f32[768] = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_24: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_12, dtype = torch.float32);  gt_12 = None
        mul_641: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_24, 1.1111111111111112);  _to_copy_24 = None
        mul_642: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_639, mul_641);  mul_641 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_409: f32[8192, 768] = torch.ops.aten.view.default(mul_642, [8192, 768]);  mul_642 = None
        mm_100: f32[8192, 3072] = torch.ops.aten.mm.default(view_409, permute_406);  permute_406 = None
        permute_407: f32[768, 8192] = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_101: f32[768, 3072] = torch.ops.aten.mm.default(permute_407, view_66);  permute_407 = view_66 = None
        permute_408: f32[3072, 768] = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
        sum_145: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
        view_410: f32[768] = torch.ops.aten.view.default(sum_145, [768]);  sum_145 = None
        view_411: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_100, [64, 128, 3072]);  mm_100 = None
        permute_409: f32[768, 3072] = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_659: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_411, add_270);  view_411 = add_270 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_412: f32[8192, 3072] = torch.ops.aten.view.default(mul_659, [8192, 3072]);  mul_659 = None
        mm_102: f32[8192, 768] = torch.ops.aten.mm.default(view_412, permute_410);  permute_410 = None
        permute_411: f32[3072, 8192] = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_103: f32[3072, 768] = torch.ops.aten.mm.default(permute_411, view_64);  permute_411 = view_64 = None
        permute_412: f32[768, 3072] = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        sum_146: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        view_413: f32[3072] = torch.ops.aten.view.default(sum_146, [3072]);  sum_146 = None
        view_414: f32[64, 128, 768] = torch.ops.aten.view.default(mm_102, [64, 128, 768]);  mm_102 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_271: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_639, view_414);  mul_639 = view_414 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_413: f32[3072, 768] = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_661: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_271, primals_62);  primals_62 = None
        mul_662: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_661, 768)
        sum_147: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_661, [2], True)
        mul_663: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_661, mul_78);  mul_661 = None
        sum_148: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_663, [2], True);  mul_663 = None
        mul_664: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_78, sum_148);  sum_148 = None
        sub_128: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_662, sum_147);  mul_662 = sum_147 = None
        sub_129: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_128, mul_664);  sub_128 = mul_664 = None
        mul_665: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_51, sub_129);  div_51 = sub_129 = None
        mul_666: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_271, mul_78);  mul_78 = None
        sum_149: f32[768] = torch.ops.aten.sum.dim_IntList(mul_666, [0, 1]);  mul_666 = None
        sum_150: f32[768] = torch.ops.aten.sum.dim_IntList(add_271, [0, 1]);  add_271 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_25: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_11, dtype = torch.float32);  gt_11 = None
        mul_667: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_25, 1.1111111111111112);  _to_copy_25 = None
        mul_668: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_665, mul_667);  mul_667 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_415: f32[8192, 768] = torch.ops.aten.view.default(mul_668, [8192, 768]);  mul_668 = None
        mm_104: f32[8192, 768] = torch.ops.aten.mm.default(view_415, permute_414);  permute_414 = None
        permute_415: f32[768, 8192] = torch.ops.aten.permute.default(view_415, [1, 0])
        mm_105: f32[768, 768] = torch.ops.aten.mm.default(permute_415, view_62);  permute_415 = view_62 = None
        permute_416: f32[768, 768] = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        sum_151: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
        view_416: f32[768] = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
        view_417: f32[64, 128, 768] = torch.ops.aten.view.default(mm_104, [64, 128, 768]);  mm_104 = None
        permute_417: f32[768, 768] = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_418: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_417, [64, 128, 12, 64]);  view_417 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_418: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_80: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        _unsafe_view_92: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_80, [768, 128, 64]);  clone_80 = None
        bmm_56: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_419, _unsafe_view_92);  permute_419 = None
        bmm_57: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_92, permute_420);  _unsafe_view_92 = permute_420 = None
        view_419: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_56, [64, 12, 128, 64]);  bmm_56 = None
        view_420: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_57, [64, 12, 128, 128]);  bmm_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_26: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_10, dtype = torch.float32);  gt_10 = None
        mul_669: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_26, 1.1111111111111112);  _to_copy_26 = None
        mul_670: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_420, mul_669);  view_420 = mul_669 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_671: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_670, alias_99);  mul_670 = None
        sum_152: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_671, [-1], True)
        mul_672: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_99, sum_152);  alias_99 = sum_152 = None
        sub_130: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_52: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_130, 8.0);  sub_130 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_421: f32[768, 128, 128] = torch.ops.aten.view.default(div_52, [768, 128, 128]);  div_52 = None
        bmm_58: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_421, view_421);  permute_421 = None
        bmm_59: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_421, permute_422);  view_421 = permute_422 = None
        view_422: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_58, [64, 12, 64, 128]);  bmm_58 = None
        view_423: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_59, [64, 12, 128, 64]);  bmm_59 = None
        permute_423: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_422, [0, 1, 3, 2]);  view_422 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_424: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_81: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        _unsafe_view_93: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_81, [64, 128, 768]);  clone_81 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_425: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_82: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        _unsafe_view_94: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_82, [64, 128, 768]);  clone_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_424: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_94, [8192, 768]);  _unsafe_view_94 = None
        mm_106: f32[8192, 768] = torch.ops.aten.mm.default(view_424, permute_426);  permute_426 = None
        permute_427: f32[768, 8192] = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_107: f32[768, 768] = torch.ops.aten.mm.default(permute_427, view_51);  permute_427 = None
        permute_428: f32[768, 768] = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        sum_153: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_424, [0], True);  view_424 = None
        view_425: f32[768] = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
        view_426: f32[64, 128, 768] = torch.ops.aten.view.default(mm_106, [64, 128, 768]);  mm_106 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_272: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_665, view_426);  mul_665 = view_426 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_429: f32[768, 768] = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_430: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_427: f32[64, 128, 768] = torch.ops.aten.view.default(permute_430, [64, 128, 768]);  permute_430 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_83: f32[64, 128, 768] = torch.ops.aten.clone.default(view_427, memory_format = torch.contiguous_format);  view_427 = None
        _unsafe_view_95: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_83, [8192, 768]);  clone_83 = None
        mm_108: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_95, permute_431);  permute_431 = None
        permute_432: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_95, [1, 0])
        mm_109: f32[768, 768] = torch.ops.aten.mm.default(permute_432, view_51);  permute_432 = None
        permute_433: f32[768, 768] = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
        sum_154: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_95, [0], True);  _unsafe_view_95 = None
        view_428: f32[768] = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
        view_429: f32[64, 128, 768] = torch.ops.aten.view.default(mm_108, [64, 128, 768]);  mm_108 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_273: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_272, view_429);  add_272 = view_429 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_434: f32[768, 768] = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_430: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_93, [8192, 768]);  _unsafe_view_93 = None
        mm_110: f32[8192, 768] = torch.ops.aten.mm.default(view_430, permute_435);  permute_435 = None
        permute_436: f32[768, 8192] = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_111: f32[768, 768] = torch.ops.aten.mm.default(permute_436, view_51);  permute_436 = view_51 = None
        permute_437: f32[768, 768] = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
        sum_155: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        view_431: f32[768] = torch.ops.aten.view.default(sum_155, [768]);  sum_155 = None
        view_432: f32[64, 128, 768] = torch.ops.aten.view.default(mm_110, [64, 128, 768]);  mm_110 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_274: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_273, view_432);  add_273 = view_432 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_438: f32[768, 768] = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_674: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_274, primals_52);  primals_52 = None
        mul_675: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_674, 768)
        sum_156: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_674, [2], True)
        mul_676: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_674, mul_72);  mul_674 = None
        sum_157: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_676, [2], True);  mul_676 = None
        mul_677: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_72, sum_157);  sum_157 = None
        sub_132: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_675, sum_156);  mul_675 = sum_156 = None
        sub_133: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_132, mul_677);  sub_132 = mul_677 = None
        mul_678: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_53, sub_133);  div_53 = sub_133 = None
        mul_679: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_274, mul_72);  mul_72 = None
        sum_158: f32[768] = torch.ops.aten.sum.dim_IntList(mul_679, [0, 1]);  mul_679 = None
        sum_159: f32[768] = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_27: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_9, dtype = torch.float32);  gt_9 = None
        mul_680: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_27, 1.1111111111111112);  _to_copy_27 = None
        mul_681: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_678, mul_680);  mul_680 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_433: f32[8192, 768] = torch.ops.aten.view.default(mul_681, [8192, 768]);  mul_681 = None
        mm_112: f32[8192, 3072] = torch.ops.aten.mm.default(view_433, permute_439);  permute_439 = None
        permute_440: f32[768, 8192] = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_113: f32[768, 3072] = torch.ops.aten.mm.default(permute_440, view_49);  permute_440 = view_49 = None
        permute_441: f32[3072, 768] = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
        sum_160: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        view_434: f32[768] = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
        view_435: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_112, [64, 128, 3072]);  mm_112 = None
        permute_442: f32[768, 3072] = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_698: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_435, add_281);  view_435 = add_281 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_436: f32[8192, 3072] = torch.ops.aten.view.default(mul_698, [8192, 3072]);  mul_698 = None
        mm_114: f32[8192, 768] = torch.ops.aten.mm.default(view_436, permute_443);  permute_443 = None
        permute_444: f32[3072, 8192] = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_115: f32[3072, 768] = torch.ops.aten.mm.default(permute_444, view_47);  permute_444 = view_47 = None
        permute_445: f32[768, 3072] = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        sum_161: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        view_437: f32[3072] = torch.ops.aten.view.default(sum_161, [3072]);  sum_161 = None
        view_438: f32[64, 128, 768] = torch.ops.aten.view.default(mm_114, [64, 128, 768]);  mm_114 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_282: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_678, view_438);  mul_678 = view_438 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_446: f32[3072, 768] = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_700: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_282, primals_46);  primals_46 = None
        mul_701: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_700, 768)
        sum_162: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_700, [2], True)
        mul_702: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_700, mul_55);  mul_700 = None
        sum_163: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_702, [2], True);  mul_702 = None
        mul_703: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_55, sum_163);  sum_163 = None
        sub_136: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_701, sum_162);  mul_701 = sum_162 = None
        sub_137: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_136, mul_703);  sub_136 = mul_703 = None
        mul_704: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_54, sub_137);  div_54 = sub_137 = None
        mul_705: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_282, mul_55);  mul_55 = None
        sum_164: f32[768] = torch.ops.aten.sum.dim_IntList(mul_705, [0, 1]);  mul_705 = None
        sum_165: f32[768] = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_28: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_8, dtype = torch.float32);  gt_8 = None
        mul_706: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_28, 1.1111111111111112);  _to_copy_28 = None
        mul_707: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_704, mul_706);  mul_706 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_439: f32[8192, 768] = torch.ops.aten.view.default(mul_707, [8192, 768]);  mul_707 = None
        mm_116: f32[8192, 768] = torch.ops.aten.mm.default(view_439, permute_447);  permute_447 = None
        permute_448: f32[768, 8192] = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_117: f32[768, 768] = torch.ops.aten.mm.default(permute_448, view_45);  permute_448 = view_45 = None
        permute_449: f32[768, 768] = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
        sum_166: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
        view_440: f32[768] = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
        view_441: f32[64, 128, 768] = torch.ops.aten.view.default(mm_116, [64, 128, 768]);  mm_116 = None
        permute_450: f32[768, 768] = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_442: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_441, [64, 128, 12, 64]);  view_441 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_451: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_84: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        _unsafe_view_96: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_84, [768, 128, 64]);  clone_84 = None
        bmm_60: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_452, _unsafe_view_96);  permute_452 = None
        bmm_61: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_96, permute_453);  _unsafe_view_96 = permute_453 = None
        view_443: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_60, [64, 12, 128, 64]);  bmm_60 = None
        view_444: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_61, [64, 12, 128, 128]);  bmm_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_29: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_7, dtype = torch.float32);  gt_7 = None
        mul_708: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_29, 1.1111111111111112);  _to_copy_29 = None
        mul_709: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_444, mul_708);  view_444 = mul_708 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_710: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_709, alias_101);  mul_709 = None
        sum_167: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_710, [-1], True)
        mul_711: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_101, sum_167);  alias_101 = sum_167 = None
        sub_138: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_55: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_138, 8.0);  sub_138 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_445: f32[768, 128, 128] = torch.ops.aten.view.default(div_55, [768, 128, 128]);  div_55 = None
        bmm_62: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_454, view_445);  permute_454 = None
        bmm_63: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_445, permute_455);  view_445 = permute_455 = None
        view_446: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_62, [64, 12, 64, 128]);  bmm_62 = None
        view_447: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_63, [64, 12, 128, 64]);  bmm_63 = None
        permute_456: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_446, [0, 1, 3, 2]);  view_446 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_457: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_85: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        _unsafe_view_97: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_85, [64, 128, 768]);  clone_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_458: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_86: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        _unsafe_view_98: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_86, [64, 128, 768]);  clone_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_448: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_98, [8192, 768]);  _unsafe_view_98 = None
        mm_118: f32[8192, 768] = torch.ops.aten.mm.default(view_448, permute_459);  permute_459 = None
        permute_460: f32[768, 8192] = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_119: f32[768, 768] = torch.ops.aten.mm.default(permute_460, view_34);  permute_460 = None
        permute_461: f32[768, 768] = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
        sum_168: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
        view_449: f32[768] = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
        view_450: f32[64, 128, 768] = torch.ops.aten.view.default(mm_118, [64, 128, 768]);  mm_118 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_283: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_704, view_450);  mul_704 = view_450 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_462: f32[768, 768] = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_463: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_451: f32[64, 128, 768] = torch.ops.aten.view.default(permute_463, [64, 128, 768]);  permute_463 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_87: f32[64, 128, 768] = torch.ops.aten.clone.default(view_451, memory_format = torch.contiguous_format);  view_451 = None
        _unsafe_view_99: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_87, [8192, 768]);  clone_87 = None
        mm_120: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_99, permute_464);  permute_464 = None
        permute_465: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_99, [1, 0])
        mm_121: f32[768, 768] = torch.ops.aten.mm.default(permute_465, view_34);  permute_465 = None
        permute_466: f32[768, 768] = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        sum_169: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_99, [0], True);  _unsafe_view_99 = None
        view_452: f32[768] = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
        view_453: f32[64, 128, 768] = torch.ops.aten.view.default(mm_120, [64, 128, 768]);  mm_120 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_284: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_283, view_453);  add_283 = view_453 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_467: f32[768, 768] = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_454: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_97, [8192, 768]);  _unsafe_view_97 = None
        mm_122: f32[8192, 768] = torch.ops.aten.mm.default(view_454, permute_468);  permute_468 = None
        permute_469: f32[768, 8192] = torch.ops.aten.permute.default(view_454, [1, 0])
        mm_123: f32[768, 768] = torch.ops.aten.mm.default(permute_469, view_34);  permute_469 = view_34 = None
        permute_470: f32[768, 768] = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
        sum_170: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
        view_455: f32[768] = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
        view_456: f32[64, 128, 768] = torch.ops.aten.view.default(mm_122, [64, 128, 768]);  mm_122 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_285: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_284, view_456);  add_284 = view_456 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_471: f32[768, 768] = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_713: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_285, primals_36);  primals_36 = None
        mul_714: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_713, 768)
        sum_171: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_713, [2], True)
        mul_715: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_713, mul_49);  mul_713 = None
        sum_172: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_715, [2], True);  mul_715 = None
        mul_716: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_49, sum_172);  sum_172 = None
        sub_140: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_714, sum_171);  mul_714 = sum_171 = None
        sub_141: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_140, mul_716);  sub_140 = mul_716 = None
        mul_717: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_56, sub_141);  div_56 = sub_141 = None
        mul_718: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_285, mul_49);  mul_49 = None
        sum_173: f32[768] = torch.ops.aten.sum.dim_IntList(mul_718, [0, 1]);  mul_718 = None
        sum_174: f32[768] = torch.ops.aten.sum.dim_IntList(add_285, [0, 1]);  add_285 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_30: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_6, dtype = torch.float32);  gt_6 = None
        mul_719: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_30, 1.1111111111111112);  _to_copy_30 = None
        mul_720: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_717, mul_719);  mul_719 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_457: f32[8192, 768] = torch.ops.aten.view.default(mul_720, [8192, 768]);  mul_720 = None
        mm_124: f32[8192, 3072] = torch.ops.aten.mm.default(view_457, permute_472);  permute_472 = None
        permute_473: f32[768, 8192] = torch.ops.aten.permute.default(view_457, [1, 0])
        mm_125: f32[768, 3072] = torch.ops.aten.mm.default(permute_473, view_32);  permute_473 = view_32 = None
        permute_474: f32[3072, 768] = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
        sum_175: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
        view_458: f32[768] = torch.ops.aten.view.default(sum_175, [768]);  sum_175 = None
        view_459: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_124, [64, 128, 3072]);  mm_124 = None
        permute_475: f32[768, 3072] = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_737: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_459, add_292);  view_459 = add_292 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_460: f32[8192, 3072] = torch.ops.aten.view.default(mul_737, [8192, 3072]);  mul_737 = None
        mm_126: f32[8192, 768] = torch.ops.aten.mm.default(view_460, permute_476);  permute_476 = None
        permute_477: f32[3072, 8192] = torch.ops.aten.permute.default(view_460, [1, 0])
        mm_127: f32[3072, 768] = torch.ops.aten.mm.default(permute_477, view_30);  permute_477 = view_30 = None
        permute_478: f32[768, 3072] = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
        sum_176: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_460, [0], True);  view_460 = None
        view_461: f32[3072] = torch.ops.aten.view.default(sum_176, [3072]);  sum_176 = None
        view_462: f32[64, 128, 768] = torch.ops.aten.view.default(mm_126, [64, 128, 768]);  mm_126 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_293: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_717, view_462);  mul_717 = view_462 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_479: f32[3072, 768] = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_739: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_293, primals_30);  primals_30 = None
        mul_740: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_739, 768)
        sum_177: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_739, [2], True)
        mul_741: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_739, mul_32);  mul_739 = None
        sum_178: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_741, [2], True);  mul_741 = None
        mul_742: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_32, sum_178);  sum_178 = None
        sub_144: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_740, sum_177);  mul_740 = sum_177 = None
        sub_145: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_144, mul_742);  sub_144 = mul_742 = None
        mul_743: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_57, sub_145);  div_57 = sub_145 = None
        mul_744: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_293, mul_32);  mul_32 = None
        sum_179: f32[768] = torch.ops.aten.sum.dim_IntList(mul_744, [0, 1]);  mul_744 = None
        sum_180: f32[768] = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_31: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_5, dtype = torch.float32);  gt_5 = None
        mul_745: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_31, 1.1111111111111112);  _to_copy_31 = None
        mul_746: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_743, mul_745);  mul_745 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_463: f32[8192, 768] = torch.ops.aten.view.default(mul_746, [8192, 768]);  mul_746 = None
        mm_128: f32[8192, 768] = torch.ops.aten.mm.default(view_463, permute_480);  permute_480 = None
        permute_481: f32[768, 8192] = torch.ops.aten.permute.default(view_463, [1, 0])
        mm_129: f32[768, 768] = torch.ops.aten.mm.default(permute_481, view_28);  permute_481 = view_28 = None
        permute_482: f32[768, 768] = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        sum_181: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
        view_464: f32[768] = torch.ops.aten.view.default(sum_181, [768]);  sum_181 = None
        view_465: f32[64, 128, 768] = torch.ops.aten.view.default(mm_128, [64, 128, 768]);  mm_128 = None
        permute_483: f32[768, 768] = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_466: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_465, [64, 128, 12, 64]);  view_465 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_484: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_88: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        _unsafe_view_100: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_88, [768, 128, 64]);  clone_88 = None
        bmm_64: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_485, _unsafe_view_100);  permute_485 = None
        bmm_65: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_100, permute_486);  _unsafe_view_100 = permute_486 = None
        view_467: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_64, [64, 12, 128, 64]);  bmm_64 = None
        view_468: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_65, [64, 12, 128, 128]);  bmm_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_32: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_4, dtype = torch.float32);  gt_4 = None
        mul_747: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_32, 1.1111111111111112);  _to_copy_32 = None
        mul_748: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_468, mul_747);  view_468 = mul_747 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_749: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_748, alias_103);  mul_748 = None
        sum_182: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_749, [-1], True)
        mul_750: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_103, sum_182);  alias_103 = sum_182 = None
        sub_146: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_749, mul_750);  mul_749 = mul_750 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_58: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_146, 8.0);  sub_146 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_469: f32[768, 128, 128] = torch.ops.aten.view.default(div_58, [768, 128, 128]);  div_58 = None
        bmm_66: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_487, view_469);  permute_487 = None
        bmm_67: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_469, permute_488);  view_469 = permute_488 = None
        view_470: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_66, [64, 12, 64, 128]);  bmm_66 = None
        view_471: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_67, [64, 12, 128, 64]);  bmm_67 = None
        permute_489: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_470, [0, 1, 3, 2]);  view_470 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_490: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_89: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        _unsafe_view_101: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_89, [64, 128, 768]);  clone_89 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_491: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_90: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        _unsafe_view_102: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_90, [64, 128, 768]);  clone_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_472: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_102, [8192, 768]);  _unsafe_view_102 = None
        mm_130: f32[8192, 768] = torch.ops.aten.mm.default(view_472, permute_492);  permute_492 = None
        permute_493: f32[768, 8192] = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_131: f32[768, 768] = torch.ops.aten.mm.default(permute_493, view_17);  permute_493 = None
        permute_494: f32[768, 768] = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
        sum_183: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        view_473: f32[768] = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
        view_474: f32[64, 128, 768] = torch.ops.aten.view.default(mm_130, [64, 128, 768]);  mm_130 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_294: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_743, view_474);  mul_743 = view_474 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_495: f32[768, 768] = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_496: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_475: f32[64, 128, 768] = torch.ops.aten.view.default(permute_496, [64, 128, 768]);  permute_496 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_91: f32[64, 128, 768] = torch.ops.aten.clone.default(view_475, memory_format = torch.contiguous_format);  view_475 = None
        _unsafe_view_103: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_91, [8192, 768]);  clone_91 = None
        mm_132: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_103, permute_497);  permute_497 = None
        permute_498: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_103, [1, 0])
        mm_133: f32[768, 768] = torch.ops.aten.mm.default(permute_498, view_17);  permute_498 = None
        permute_499: f32[768, 768] = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
        sum_184: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_103, [0], True);  _unsafe_view_103 = None
        view_476: f32[768] = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
        view_477: f32[64, 128, 768] = torch.ops.aten.view.default(mm_132, [64, 128, 768]);  mm_132 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_295: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_294, view_477);  add_294 = view_477 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_500: f32[768, 768] = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_478: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_101, [8192, 768]);  _unsafe_view_101 = None
        mm_134: f32[8192, 768] = torch.ops.aten.mm.default(view_478, permute_501);  permute_501 = None
        permute_502: f32[768, 8192] = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_135: f32[768, 768] = torch.ops.aten.mm.default(permute_502, view_17);  permute_502 = view_17 = None
        permute_503: f32[768, 768] = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        sum_185: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        view_479: f32[768] = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
        view_480: f32[64, 128, 768] = torch.ops.aten.view.default(mm_134, [64, 128, 768]);  mm_134 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_296: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_295, view_480);  add_295 = view_480 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_504: f32[768, 768] = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_752: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_296, primals_20);  primals_20 = None
        mul_753: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_752, 768)
        sum_186: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_752, [2], True)
        mul_754: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_752, mul_26);  mul_752 = None
        sum_187: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_754, [2], True);  mul_754 = None
        mul_755: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_26, sum_187);  sum_187 = None
        sub_148: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_753, sum_186);  mul_753 = sum_186 = None
        sub_149: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_148, mul_755);  sub_148 = mul_755 = None
        mul_756: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_59, sub_149);  div_59 = sub_149 = None
        mul_757: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_296, mul_26);  mul_26 = None
        sum_188: f32[768] = torch.ops.aten.sum.dim_IntList(mul_757, [0, 1]);  mul_757 = None
        sum_189: f32[768] = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        _to_copy_33: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_3, dtype = torch.float32);  gt_3 = None
        mul_758: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_33, 1.1111111111111112);  _to_copy_33 = None
        mul_759: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_756, mul_758);  mul_758 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        view_481: f32[8192, 768] = torch.ops.aten.view.default(mul_759, [8192, 768]);  mul_759 = None
        mm_136: f32[8192, 3072] = torch.ops.aten.mm.default(view_481, permute_505);  permute_505 = None
        permute_506: f32[768, 8192] = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_137: f32[768, 3072] = torch.ops.aten.mm.default(permute_506, view_15);  permute_506 = view_15 = None
        permute_507: f32[3072, 768] = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
        sum_190: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        view_482: f32[768] = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
        view_483: f32[64, 128, 3072] = torch.ops.aten.view.default(mm_136, [64, 128, 3072]);  mm_136 = None
        permute_508: f32[768, 3072] = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_776: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_483, add_303);  view_483 = add_303 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        view_484: f32[8192, 3072] = torch.ops.aten.view.default(mul_776, [8192, 3072]);  mul_776 = None
        mm_138: f32[8192, 768] = torch.ops.aten.mm.default(view_484, permute_509);  permute_509 = None
        permute_510: f32[3072, 8192] = torch.ops.aten.permute.default(view_484, [1, 0])
        mm_139: f32[3072, 768] = torch.ops.aten.mm.default(permute_510, view_13);  permute_510 = view_13 = None
        permute_511: f32[768, 3072] = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
        sum_191: f32[1, 3072] = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
        view_485: f32[3072] = torch.ops.aten.view.default(sum_191, [3072]);  sum_191 = None
        view_486: f32[64, 128, 768] = torch.ops.aten.view.default(mm_138, [64, 128, 768]);  mm_138 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        add_304: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_756, view_486);  mul_756 = view_486 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_512: f32[3072, 768] = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_778: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_304, primals_14);  primals_14 = None
        mul_779: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_778, 768)
        sum_192: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_778, [2], True)
        mul_780: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_778, mul_9);  mul_778 = None
        sum_193: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_780, [2], True);  mul_780 = None
        mul_781: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_9, sum_193);  sum_193 = None
        sub_152: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_779, sum_192);  mul_779 = sum_192 = None
        sub_153: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_152, mul_781);  sub_152 = mul_781 = None
        mul_782: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_60, sub_153);  div_60 = sub_153 = None
        mul_783: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_304, mul_9);  mul_9 = None
        sum_194: f32[768] = torch.ops.aten.sum.dim_IntList(mul_783, [0, 1]);  mul_783 = None
        sum_195: f32[768] = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        _to_copy_34: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt_2, dtype = torch.float32);  gt_2 = None
        mul_784: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_34, 1.1111111111111112);  _to_copy_34 = None
        mul_785: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_782, mul_784);  mul_784 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        view_487: f32[8192, 768] = torch.ops.aten.view.default(mul_785, [8192, 768]);  mul_785 = None
        mm_140: f32[8192, 768] = torch.ops.aten.mm.default(view_487, permute_513);  permute_513 = None
        permute_514: f32[768, 8192] = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_141: f32[768, 768] = torch.ops.aten.mm.default(permute_514, view_11);  permute_514 = view_11 = None
        permute_515: f32[768, 768] = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
        sum_196: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
        view_488: f32[768] = torch.ops.aten.view.default(sum_196, [768]);  sum_196 = None
        view_489: f32[64, 128, 768] = torch.ops.aten.view.default(mm_140, [64, 128, 768]);  mm_140 = None
        permute_516: f32[768, 768] = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_490: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_489, [64, 128, 12, 64]);  view_489 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_517: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_92: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        _unsafe_view_104: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_92, [768, 128, 64]);  clone_92 = None
        bmm_68: f32[768, 128, 64] = torch.ops.aten.bmm.default(permute_518, _unsafe_view_104);  permute_518 = None
        bmm_69: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_104, permute_519);  _unsafe_view_104 = permute_519 = None
        view_491: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_68, [64, 12, 128, 64]);  bmm_68 = None
        view_492: f32[64, 12, 128, 128] = torch.ops.aten.view.default(bmm_69, [64, 12, 128, 128]);  bmm_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        _to_copy_35: f32[64, 12, 128, 128] = torch.ops.aten._to_copy.default(gt_1, dtype = torch.float32);  gt_1 = None
        mul_786: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_35, 1.1111111111111112);  _to_copy_35 = None
        mul_787: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(view_492, mul_786);  view_492 = mul_786 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_788: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_787, alias_105);  mul_787 = None
        sum_197: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_788, [-1], True)
        mul_789: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(alias_105, sum_197);  alias_105 = sum_197 = None
        sub_154: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_61: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(sub_154, 8.0);  sub_154 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_493: f32[768, 128, 128] = torch.ops.aten.view.default(div_61, [768, 128, 128]);  div_61 = None
        bmm_70: f32[768, 64, 128] = torch.ops.aten.bmm.default(permute_520, view_493);  permute_520 = None
        bmm_71: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_493, permute_521);  view_493 = permute_521 = None
        view_494: f32[64, 12, 64, 128] = torch.ops.aten.view.default(bmm_70, [64, 12, 64, 128]);  bmm_70 = None
        view_495: f32[64, 12, 128, 64] = torch.ops.aten.view.default(bmm_71, [64, 12, 128, 64]);  bmm_71 = None
        permute_522: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2]);  view_494 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_523: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_93: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        _unsafe_view_105: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_93, [64, 128, 768]);  clone_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_524: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        clone_94: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        _unsafe_view_106: f32[64, 128, 768] = torch.ops.aten._unsafe_view.default(clone_94, [64, 128, 768]);  clone_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_496: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_106, [8192, 768]);  _unsafe_view_106 = None
        mm_142: f32[8192, 768] = torch.ops.aten.mm.default(view_496, permute_525);  permute_525 = None
        permute_526: f32[768, 8192] = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_143: f32[768, 768] = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        permute_527: f32[768, 768] = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        sum_198: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        view_497: f32[768] = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
        view_498: f32[64, 128, 768] = torch.ops.aten.view.default(mm_142, [64, 128, 768]);  mm_142 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        add_305: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_782, view_498);  mul_782 = view_498 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_528: f32[768, 768] = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_529: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_499: f32[64, 128, 768] = torch.ops.aten.view.default(permute_529, [64, 128, 768]);  permute_529 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        clone_95: f32[64, 128, 768] = torch.ops.aten.clone.default(view_499, memory_format = torch.contiguous_format);  view_499 = None
        _unsafe_view_107: f32[8192, 768] = torch.ops.aten._unsafe_view.default(clone_95, [8192, 768]);  clone_95 = None
        mm_144: f32[8192, 768] = torch.ops.aten.mm.default(_unsafe_view_107, permute_530);  permute_530 = None
        permute_531: f32[768, 8192] = torch.ops.aten.permute.default(_unsafe_view_107, [1, 0])
        mm_145: f32[768, 768] = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
        permute_532: f32[768, 768] = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        sum_199: f32[1, 768] = torch.ops.aten.sum.dim_IntList(_unsafe_view_107, [0], True);  _unsafe_view_107 = None
        view_500: f32[768] = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
        view_501: f32[64, 128, 768] = torch.ops.aten.view.default(mm_144, [64, 128, 768]);  mm_144 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        add_306: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_305, view_501);  add_305 = view_501 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_533: f32[768, 768] = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        view_502: f32[8192, 768] = torch.ops.aten.view.default(_unsafe_view_105, [8192, 768]);  _unsafe_view_105 = None
        mm_146: f32[8192, 768] = torch.ops.aten.mm.default(view_502, permute_534);  permute_534 = None
        permute_535: f32[768, 8192] = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_147: f32[768, 768] = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        permute_536: f32[768, 768] = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
        sum_200: f32[1, 768] = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
        view_503: f32[768] = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
        view_504: f32[64, 128, 768] = torch.ops.aten.view.default(mm_146, [64, 128, 768]);  mm_146 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        add_307: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add_306, view_504);  add_306 = view_504 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_537: f32[768, 768] = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.dropout(embeddings)
        _to_copy_36: f32[64, 128, 768] = torch.ops.aten._to_copy.default(gt, dtype = torch.float32);  gt = None
        mul_790: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(_to_copy_36, 1.1111111111111112);  _to_copy_36 = None
        mul_791: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_307, mul_790);  add_307 = mul_790 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings = self.LayerNorm(embeddings)
        mul_793: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_791, primals_4);  primals_4 = None
        mul_794: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_793, 768)
        sum_201: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_793, [2], True)
        mul_795: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_793, mul_1);  mul_793 = None
        sum_202: f32[64, 128, 1] = torch.ops.aten.sum.dim_IntList(mul_795, [2], True);  mul_795 = None
        mul_796: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_1, sum_202);  sum_202 = None
        sub_156: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_794, sum_201);  mul_794 = sum_201 = None
        sub_157: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(sub_156, mul_796);  sub_156 = mul_796 = None
        mul_797: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(div_62, sub_157);  div_62 = sub_157 = None
        mul_798: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_791, mul_1);  mul_1 = None
        sum_203: f32[768] = torch.ops.aten.sum.dim_IntList(mul_798, [0, 1]);  mul_798 = None
        sum_204: f32[768] = torch.ops.aten.sum.dim_IntList(mul_791, [0, 1]);  mul_791 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: embeddings += position_embeddings
        sum_205: f32[1, 128, 768] = torch.ops.aten.sum.dim_IntList(mul_797, [0], True)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:236, code: position_embeddings = self.position_embeddings(position_ids)
        view_505: f32[128, 768] = torch.ops.aten.view.default(sum_205, [128, 768])
        new_zeros: f32[512, 768] = torch.ops.aten.new_zeros.default(sum_205, [512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sum_205 = None
        ne: b8[128] = torch.ops.aten.ne.Scalar(view_506, -1)
        unsqueeze_4: b8[128, 1] = torch.ops.aten.unsqueeze.default(ne, 1);  ne = None
        expand_49: b8[128, 768] = torch.ops.aten.expand.default(unsqueeze_4, [128, 768]);  unsqueeze_4 = None
        full_like: f32[128, 768] = torch.ops.aten.full_like.default(view_505, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_106: f32[128, 768] = torch.ops.aten.alias.default(full_like);  full_like = None
        where: f32[128, 768] = torch.ops.aten.where.self(expand_49, view_505, alias_106);  expand_49 = view_505 = alias_106 = None
        index_put: f32[512, 768] = torch.ops.aten.index_put.default(new_zeros, [view_506], where, True);  new_zeros = view_506 = where = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        view_507: f32[8192, 768] = torch.ops.aten.view.default(mul_797, [8192, 768])
        new_zeros_1: f32[2, 768] = torch.ops.aten.new_zeros.default(mul_797, [2, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_96: i64[64, 128] = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view_108: i64[8192] = torch.ops.aten._unsafe_view.default(clone_96, [8192]);  clone_96 = None
        ne_1: b8[8192] = torch.ops.aten.ne.Scalar(_unsafe_view_108, -1)
        unsqueeze_5: b8[8192, 1] = torch.ops.aten.unsqueeze.default(ne_1, 1);  ne_1 = None
        expand_50: b8[8192, 768] = torch.ops.aten.expand.default(unsqueeze_5, [8192, 768]);  unsqueeze_5 = None
        full_like_1: f32[8192, 768] = torch.ops.aten.full_like.default(view_507, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_107: f32[8192, 768] = torch.ops.aten.alias.default(full_like_1);  full_like_1 = None
        where_1: f32[8192, 768] = torch.ops.aten.where.self(expand_50, view_507, alias_107);  expand_50 = None
        index_put_1: f32[2, 768] = torch.ops.aten.index_put.default(new_zeros_1, [_unsafe_view_108], where_1, True);  new_zeros_1 = _unsafe_view_108 = where_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:231, code: inputs_embeds = self.word_embeddings(input_ids)
        new_zeros_2: f32[30522, 768] = torch.ops.aten.new_zeros.default(mul_797, [30522, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  mul_797 = None
        ne_2: b8[8192] = torch.ops.aten.ne.Scalar(view_509, 0)
        unsqueeze_6: b8[8192, 1] = torch.ops.aten.unsqueeze.default(ne_2, 1);  ne_2 = None
        expand_51: b8[8192, 768] = torch.ops.aten.expand.default(unsqueeze_6, [8192, 768]);  unsqueeze_6 = None
        where_2: f32[8192, 768] = torch.ops.aten.where.self(expand_51, view_507, alias_107);  expand_51 = view_507 = alias_107 = None
        index_put_2: f32[30522, 768] = torch.ops.aten.index_put.default(new_zeros_2, [view_509], where_2, True);  new_zeros_2 = view_509 = where_2 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:231, code: inputs_embeds = self.word_embeddings(input_ids)
        add_308: f32[30522, 768] = torch.ops.aten.add.Tensor(permute_137, index_put_2);  permute_137 = index_put_2 = None
        return [add_308, index_put_1, index_put, sum_203, sum_204, permute_537, view_503, permute_533, view_500, permute_528, view_497, permute_516, view_488, sum_194, sum_195, permute_512, view_485, permute_508, view_482, sum_188, sum_189, permute_504, view_479, permute_500, view_476, permute_495, view_473, permute_483, view_464, sum_179, sum_180, permute_479, view_461, permute_475, view_458, sum_173, sum_174, permute_471, view_455, permute_467, view_452, permute_462, view_449, permute_450, view_440, sum_164, sum_165, permute_446, view_437, permute_442, view_434, sum_158, sum_159, permute_438, view_431, permute_434, view_428, permute_429, view_425, permute_417, view_416, sum_149, sum_150, permute_413, view_413, permute_409, view_410, sum_143, sum_144, permute_405, view_407, permute_401, view_404, permute_396, view_401, permute_384, view_392, sum_134, sum_135, permute_380, view_389, permute_376, view_386, sum_128, sum_129, permute_372, view_383, permute_368, view_380, permute_363, view_377, permute_351, view_368, sum_119, sum_120, permute_347, view_365, permute_343, view_362, sum_113, sum_114, permute_339, view_359, permute_335, view_356, permute_330, view_353, permute_318, view_344, sum_104, sum_105, permute_314, view_341, permute_310, view_338, sum_98, sum_99, permute_306, view_335, permute_302, view_332, permute_297, view_329, permute_285, view_320, sum_89, sum_90, permute_281, view_317, permute_277, view_314, sum_83, sum_84, permute_273, view_311, permute_269, view_308, permute_264, view_305, permute_252, view_296, sum_74, sum_75, permute_248, view_293, permute_244, view_290, sum_68, sum_69, permute_240, view_287, permute_236, view_284, permute_231, view_281, permute_219, view_272, sum_59, sum_60, permute_215, view_269, permute_211, view_266, sum_53, sum_54, permute_207, view_263, permute_203, view_260, permute_198, view_257, permute_186, view_248, sum_44, sum_45, permute_182, view_245, permute_178, view_242, sum_38, sum_39, permute_174, view_239, permute_170, view_236, permute_165, view_233, permute_153, view_224, sum_29, sum_30, permute_149, view_221, permute_145, view_218, sum_23, sum_24, permute_141, view_215, sum_18, sum_19, view_212, None, None, None, None]
        