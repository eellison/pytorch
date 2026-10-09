import time; t=time.time()
import torch; t1=time.time()
import vllm; from vllm import LLM; t2=time.time()
print("torch %.1f s, vllm %.1f s" % (t1-t, t2-t1))
