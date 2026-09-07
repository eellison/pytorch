import torch, functools
from quack.blockscaled import quantize as qz
x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
for label, fn in (("to_mxfp4_byte", qz.to_mxfp4_byte), ("to_mx", qz.to_mx), ("to_mxfp6_e2m3_packed", qz.to_mxfp6_e2m3_packed),
                  ("to_mxfp4_byte partial(32)", functools.partial(qz.to_mxfp4_byte, block_size=32))):
    torch._dynamo.reset()
    from torch._dynamo.backends.debugging import ExplainWithBackend
    seen = {}
    def backend(gm, example_inputs):
        seen["ph"] = [(n.name, str(n.meta.get("example_value"))[:40]) for n in gm.graph.nodes if n.op == "placeholder"]
        return gm.forward
    torch.compile(fn, backend=backend, dynamic=True)(x)
    print(f"{label:28s} placeholders: {seen['ph']}")
