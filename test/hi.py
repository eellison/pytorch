import yaml
import torch
import sys
import ruamel.yaml

yaml = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given

pointwise_ops = [
    # please keep the entries below alphabetically sorted
    "aten.abs.default",
    "aten.acos.default",
    "aten.acos.out",
    "aten.acos_.default",
    "aten.acosh.default",
    "aten.acosh.out",
    "aten.acosh_.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.add.out",
    "aten.add_.Scalar",
    "aten.add_.Tensor",
    "aten.addcdiv.default",
    "aten.addcdiv.out",
    "aten.addcdiv_.default",
    "aten.addcmul.default",
    "aten.addcmul.out",
    "aten.addcmul_.default",
    "aten.angle.default",
    "aten.angle.out",
    "aten.asin.default",
    "aten.asin.out",
    "aten.asin_.default",
    "aten.asinh.default",
    "aten.asinh.out",
    "aten.asinh_.default",
    "aten.atan.default",
    "aten.atan.out",
    "aten.atan2.default",
    "aten.atan2.out",
    "aten.atan2_.default",
    "aten.atan_.default",
    "aten.atanh.default",
    "aten.atanh.out",
    "aten.atanh_.default",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and.Scalar_Tensor",
    "aten.bitwise_and.Scalar_out",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_and.Tensor_out",
    "aten.bitwise_and_.Scalar",
    "aten.bitwise_and_.Tensor",
    "aten.bitwise_left_shift.Scalar_Tensor",
    "aten.bitwise_left_shift.Tensor",
    "aten.bitwise_left_shift.Tensor_Scalar",
    "aten.bitwise_left_shift.Tensor_Scalar_out",
    "aten.bitwise_left_shift.Tensor_out",
    "aten.bitwise_left_shift_.Tensor",
    "aten.bitwise_left_shift_.Tensor_Scalar",
    "aten.bitwise_not.default",
    "aten.bitwise_not.out",
    "aten.bitwise_not_.default",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Scalar_Tensor",
    "aten.bitwise_or.Scalar_out",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_or.Tensor_out",
    "aten.bitwise_or_.Scalar",
    "aten.bitwise_or_.Tensor",
    "aten.bitwise_right_shift.Scalar_Tensor",
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_right_shift.Tensor_Scalar",
    "aten.bitwise_right_shift.Tensor_Scalar_out",
    "aten.bitwise_right_shift.Tensor_out",
    "aten.bitwise_right_shift_.Tensor",
    "aten.bitwise_right_shift_.Tensor_Scalar",
    "aten.bitwise_xor.Scalar",
    "aten.bitwise_xor.Scalar_Tensor",
    "aten.bitwise_xor.Scalar_out",
    "aten.bitwise_xor.Tensor",
    "aten.bitwise_xor.Tensor_out",
    "aten.bitwise_xor_.Scalar",
    "aten.bitwise_xor_.Tensor",
    "aten.ceil.default",
    "aten.ceil.out",
    "aten.ceil_.default",
    "aten.clamp.default",
    "aten.clamp.out",
    "aten.clamp_.default",
    "aten.clip.default",
    "aten.clip.out",
    "aten.clip_.default",
    "aten.conj_physical.default",
    "aten.conj_physical.out",
    "aten.conj_physical_.default",
    "aten.copy_sign.Scalar",
    "aten.copy_sign.Scalar_out",
    "aten.copy_sign.Tensor",
    "aten.copy_sign.out",
    "aten.copy_sign_.Scalar",
    "aten.copy_sign_.Tensor",
    "aten.cos.default",
    "aten.cos.out",
    "aten.cos_.default",
    "aten.cosh.default",
    "aten.cosh.out",
    "aten.cosh_.default",
    "aten.deg2rad.default",
    "aten.deg2rad.out",
    "aten.deg2rad_.default",
    "aten.digamma.default",
    "aten.digamma.out",
    "aten.digamma_.default",
    "aten.div.Tensor",
    "aten.div.Tensor_mode",
    "aten.div.out",
    "aten.div.out_mode",
    "aten.div_.Tensor",
    "aten.div_.Tensor_mode",
    "aten.eq.Tensor",
    "aten.eq.Tensor_out",
    "aten.eq.Scalar",
    "aten.eq.Scalar_out",
    "aten.equal.default",
    "aten.erf.default",
    "aten.erf.out",
    "aten.erf_.default",
    "aten.erfc.default",
    "aten.erfc.out",
    "aten.erfc_.default",
    "aten.erfinv.default",
    "aten.erfinv.out",
    "aten.erfinv_.default",
    "aten.exp.default",
    "aten.exp.out",
    "aten.exp2.default",
    "aten.exp2.out",
    "aten.exp2_.default",
    "aten.exp_.default",
    "aten.expm1.default",
    "aten.expm1.out",
    "aten.expm1_.default",
    "aten.float_power.Scalar",
    "aten.float_power.Scalar_out",
    "aten.float_power.Tensor_Scalar",
    "aten.float_power.Tensor_Scalar_out",
    "aten.float_power.Tensor_Tensor",
    "aten.float_power.Tensor_Tensor_out",
    "aten.float_power_.Scalar",
    "aten.float_power_.Tensor",
    "aten.floor.default",
    "aten.floor.out",
    "aten.floor_.default",
    "aten.fmod.Scalar",
    "aten.fmod.Scalar_out",
    "aten.fmod.Tensor",
    "aten.fmod.Tensor_out",
    "aten.fmod_.Scalar",
    "aten.fmod_.Tensor",
    "aten.frac.default",
    "aten.frac.out",
    "aten.frac_.default",
    "aten.ge.Scalar",
    "aten.ge.Tensor",
    "aten.gelu.default",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.hypot.default",
    "aten.hypot.out",
    "aten.hypot_.default",
    "aten.i0.default",
    "aten.i0.out",
    "aten.i0_.default",
    "aten.igamma.default",
    "aten.igamma.out",
    "aten.igamma_.default",
    "aten.igammac.default",
    "aten.igammac.out",
    "aten.igammac_.default",
    "aten.isnan.default",
    "aten.ldexp.default",
    "aten.ldexp.out",
    "aten.ldexp_.default",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.lerp.Scalar",
    "aten.lerp.Scalar_out",
    "aten.lerp.Tensor",
    "aten.lerp.Tensor_out",
    "aten.lerp_.Scalar",
    "aten.lerp_.Tensor",
    "aten.lgamma.default",
    "aten.lgamma.out",
    "aten.lgamma_.default",
    "aten.log.default",
    "aten.log.out",
    "aten.log10.default",
    "aten.log10.out",
    "aten.log10_.default",
    "aten.log1p.default",
    "aten.log1p.out",
    "aten.log1p_.default",
    "aten.log2.default",
    "aten.log2.out",
    "aten.log2_.default",
    "aten.log_.default",
    "aten.logaddexp.default",
    "aten.logaddexp.out",
    "aten.logaddexp2.default",
    "aten.logaddexp2.out",
    "aten.logical_and.default",
    "aten.logical_and.out",
    "aten.logical_and_.default",
    "aten.logical_not.default",
    "aten.logical_not.out",
    "aten.logical_not_.default",
    "aten.logical_or.default",
    "aten.logical_or.out",
    "aten.logical_or_.default",
    "aten.logical_xor.default",
    "aten.logical_xor.out",
    "aten.logical_xor_.default",
    "aten.logit.default",
    "aten.logit.out",
    "aten.logit_.default",
    "aten.masked_fill.Scalar",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.mul.out",
    "aten.mul_.Scalar",
    "aten.mul_.Tensor",
    "aten.mvlgamma.default",
    "aten.mvlgamma.out",
    "aten.mvlgamma_.default",
    "aten.native_dropout_backward.default",
    "aten.native_dropout_backward.out",
    "aten.nan_to_num.default",
    "aten.nan_to_num.out",
    "aten.nan_to_num_.default",
    "aten.ne.Scalar",
    "aten.neg.default",
    "aten.neg.out",
    "aten.neg_.default",
    "aten.nextafter.default",
    "aten.nextafter.out",
    "aten.nextafter_.default",
    "aten.polygamma.default",
    "aten.polygamma.out",
    "aten.polygamma_.default",
    "aten.positive.default",
    "aten.pow.Scalar",
    "aten.pow.Scalar_out",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Scalar_out",
    "aten.pow.Tensor_Tensor",
    "aten.pow.Tensor_Tensor_out",
    "aten.pow_.Scalar",
    "aten.pow_.Tensor",
    "aten.reciprocal.default",
    "aten.reciprocal.out",
    "aten.reciprocal_.default",
    "aten.red2deg.default",
    "aten.red2deg.out",
    "aten.red2deg_.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.remainder.Scalar",
    "aten.remainder.Scalar_Tensor",
    "aten.remainder.Scalar_out",
    "aten.remainder.Tensor",
    "aten.remainder.Tensor_out",
    "aten.remainder_.Scalar",
    "aten.remainder_.Tensor",
    "aten.round.decimals",
    "aten.round.decimals_out",
    "aten.round.default",
    "aten.round.out",
    "aten.round_.decimals",
    "aten.round_.default",
    "aten.rsqrt.default",
    "aten.rsqrt.out",
    "aten.rsqrt_.default",
    "aten.rsub.Scalar",
    "aten.sgn.default",
    "aten.sgn.out",
    "aten.sgn_.default",
    "aten.sigmoid.default",
    "aten.sigmoid.out",
    "aten.sigmoid_.default",
    "aten.sign.default",
    "aten.sign.out",
    "aten.sign_.default",
    "aten.signbit.default",
    "aten.signbit.out",
    "aten.sin.default",
    "aten.sin.out",
    "aten.sin_.default",
    "aten.sinc.default",
    "aten.sinc.out",
    "aten.sinc_.default",
    "aten.sinh.default",
    "aten.sinh.out",
    "aten.sinh_.default",
    "aten.sqrt.default",
    "aten.sqrt.out",
    "aten.sqrt_.default",
    "aten.square.default",
    "aten.square.out",
    "aten.square_.default",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.sub.out",
    "aten.sub_.Scalar",
    "aten.sub_.Tensor",
    "aten.tan.default",
    "aten.tan.out",
    "aten.tan_.default",
    "aten.tanh.default",
    "aten.tanh.out",
    "aten.tanh_.default",
    "aten.true_divide.Tensor",
    "aten.trunc.default",
    "aten.trunc.out",
    "aten.trunc_.default",
    "aten.where.self",
    "aten.xlogy.OutScalar_Self",
    "aten.xlogy.OutTensor",
    "aten.xlogy.Scalar_other",
    "aten.xlogy.Scalar_self",
    "aten.xlogy.Tensor",
    "aten.xlogy_.OutScalar_Other",
    "aten.xlogy_.Scalar_other",
    "aten.xlogy_.Tensor",
    "prims.convert_element_type.default",
    # backward point-wise ops
    # please keep the entries below alphabetically sorted
    "aten.gelu_backward.default",
    "aten.sigmoid_backward.default",
    "aten.tanh_backward.default",
    "aten.threshold_backward.default",
]

from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

def L(*l):
   ret = ruamel.yaml.comments.CommentedSeq(l)
   ret.fa.set_flow_style()
   return ret   


pointwise_ops = [tuple(op.split(".")[1:]) for op in pointwise_ops]

file_name = "/scratch/eellison/work/pytorch/aten/src/ATen/native/native_functions.yaml"
import ruamel
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']
yaml.width = 4096

with open(file_name, 'r') as file:
    native_funcs = yaml.load(file)


def process_entry(entry):
    signature = entry["func"]
    name = signature[:signature.find("(")]
    if "." in name:
        name, overload = name.split(".")
    else:
        overload = "default"
    
    if (name, overload) not in pointwise_ops:
        return
    
    if "tags" not in entry:
        entry["tags"] = "pointwise"
        return

    existing_tags = entry["tags"]
    if not isinstance(existing_tags, list):
        existing_tags = [existing_tags]
    entry["tags"] = L(*(existing_tags + ["pointwise"]))
    return


for iter, entry in enumerate(native_funcs):
    process_entry(entry)

new_filename = 'names.yaml'

with open(new_filename, 'w') as file:
    yaml.dump(native_funcs, file)

# annoying thing where an empty line is a comment, and so adding a new entry goes
# after the comment.
with open(new_filename, "r") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        if "tags: pointwise" in lines[i] and lines[i - 1] == "\n":
            tmp = lines[i]
            lines[i] = lines[i - 1]
            lines [i - 1] = tmp
        elif "tags: pointwise" in lines[i] and lines[i - 2] == "\n":
            tmp = lines[i]
            lines[i] = lines[i - 2]
            lines [i - 2] = tmp

with open(new_filename, "w") as file:
    for line in lines:
        file.write(line)




# breakpoint()
# print(native_funcs)