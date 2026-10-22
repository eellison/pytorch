import importlib, sys
for m in ["tvm_ffi","ninja","flashinfer","vllm","transformers","triton","cuda.bindings","nvidia.cudnn","numpy","safetensors","tokenizers","msgspec","zmq"]:
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__file__",None), getattr(mod,"__version__",""))
    except Exception as e: print(m, "ERR", e)
import flashinfer, os; d=os.path.dirname(flashinfer.__file__); print(os.listdir(d+"/data") if os.path.isdir(d+"/data") else "no data dir")
import torch; print("torch", torch.__file__, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); import torch.cuda._host_trace_replay; x=torch.ones(4,device="cuda"); print((x*2).sum().item())
