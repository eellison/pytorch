# One-time fetch (user-approved downloads, keep disk small): every sm100f trtllm-gen batched-GEMM cubin vLLM's NVFP4
# FLASHINFER_TRTLLM MoE can select (FC1: E2m1 out, swiGlu, biasFp32M; FC2: Bfloat16 out, no activation), so tactic choice
# at any M never misses. Names and sha256 come from FlashInfer's own checksums.txt; get_artifact verifies each file.
import os, re, sys
from flashinfer.jit.cubin_loader import get_artifact
from flashinfer.jit.env import FLASHINFER_CUBIN_DIR
d = "8ec29a98612c3670f9f28825d1ed19f09496073b/batched_gemm-fa419f4-31ee4e5/"
fc1 = re.compile(r"^Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_.*_biasFp32M_bN_.*swiGlu_dynB_sm100f\.cubin$")
fc2 = re.compile(r"^Bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16(tokFp32)?_.*_biasFp32M_bN_.*_dynB_sm100f\.cubin$")
want = []
for line in open(FLASHINFER_CUBIN_DIR / d / "checksums.txt"):
    sha, name = line.split()
    if fc1.match(name) or (fc2.match(name) and not re.search("swiGlu|relu2|geGlu", name)):
        want.append((name, sha))
print(len(want), "selected", FLASHINFER_CUBIN_DIR, flush=True)
new = 0
for name, sha in want:
    p = FLASHINFER_CUBIN_DIR / d / name
    had = p.exists()
    if not get_artifact(d + name, sha):
        sys.exit(f"FAILED {name}")
    new += not had
print("fetched", new, "new files")
