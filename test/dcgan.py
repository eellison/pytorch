import torch
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN")
gnet = model.getNetG()
t = torch.rand((1, 64, 8, 8))
traced_model = torch.jit.trace(gnet, [t])
import pdb; pdb.set_trace()

torch.jit.save(traced_model, "models/dcgan.pt")
