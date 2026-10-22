# Host cost (us) of the pieces of RequestState.apply_staged_writes / Sampler.apply_staged_writes and of prepare_inputs' H2D copies.
import time, numpy as np, torch
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor, async_copy_to_gpu
from vllm.utils.torch_utils import async_tensor_h2d
dev = torch.device("cuda")
def t(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    torch.cuda.synchronize(); return round(sorted(ts)[n // 2] * 1e6, 1)
toks = list(range(1000, 1000 + 2048)); toks512 = toks[:512]
R = {}
R["torch.tensor(list2048,pin)"] = t(lambda: torch.tensor(toks, dtype=torch.int32, pin_memory=True))
R["np.array(list2048)"] = t(lambda: np.array(toks, dtype=np.int32))
R["np.array(list512)"] = t(lambda: np.array(toks512, dtype=np.int32))
R["async_tensor_h2d(list2048)"] = t(lambda: async_tensor_h2d(toks, device=dev, dtype=torch.int32))
R["async_tensor_h2d(list1)"] = t(lambda: async_tensor_h2d([5], device=dev, dtype=torch.int32))
uva = UvaBackedTensor(1024, dtype=torch.int32)
R["UvaBackedTensor.copy_to_uva(1024)"] = t(lambda: uva.copy_to_uva())
big = StagedWriteTensor((64, 4096), dtype=torch.int32, device=dev, uva_instead_of_gpu=True)
def w(x):
    big.stage_write(3, 0, x); big.apply_write()
R["StagedWrite 2048 uva"] = t(lambda: w(toks))
R["StagedWrite 512 uva"] = t(lambda: w(toks512))
small = StagedWriteTensor(1024, dtype=torch.int32, device=dev)
def e():
    small.stage_write_elem(3, 7); small.apply_write()
R["StagedWrite elem"] = t(e)
R["np row assign 2048"] = t(lambda: big._uva_buf.np.__setitem__((3, slice(0, 2048)), toks))
idx = np.arange(8, dtype=np.intp); qsl = np.zeros(1025, dtype=np.int32); qbuf = torch.zeros(1025, dtype=torch.int32, device=dev)
R["async_copy_to_gpu idx(8)"] = t(lambda: async_copy_to_gpu(idx, device=dev))
R["async_copy_to_gpu qsl(1025,out)"] = t(lambda: async_copy_to_gpu(qsl, out=qbuf))
pin = torch.zeros(4096, dtype=torch.uint8, pin_memory=True); pnp = pin.numpy(); dbuf = torch.zeros(4096, dtype=torch.uint8, device=dev)
def stage():
    pnp[:64].view(np.int64)[:8] = idx; pnp[64:64 + 4100].view(np.int32)[:1025] = qsl if False else 0
    dbuf.copy_(pin, non_blocking=True)
R["one pinned stage + 1 memcpy"] = t(stage)
R["torch.arange(9,dev)"] = t(lambda: torch.arange(9, device=dev, dtype=torch.int32))
R["torch.zeros(8,dev)"] = t(lambda: torch.zeros(8, dtype=torch.int32, device=dev))
for k, v in R.items(): print(f"{v:8.1f}  {k}")
