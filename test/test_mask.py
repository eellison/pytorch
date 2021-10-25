import torch
import torchvision
model_tv = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model_tv.eval()
import pdb; pdb.set_trace()
model = torch.onnx.export(model_tv, torch.rand(1,3,300,300), "mask_rcnn_r50_fpn.onnx",
                  do_constant_folding=True,
                  opset_version=11  # opset_version 11 required for Mask R-CNN
                  )
import pdb; pdb.set_trace()
print(model)