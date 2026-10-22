# trtllm_cpp: FlashInfer's trtllm-gen launcher traced through its own C++ -- how a run gets it (2026-10-08)

Everything lives in land/core/trtllm_cpp, outside any torch tree. It needs no torch change and no rebuild of
FlashInfer's own modules: FlashInfer's fmha_gen stays the stock JIT build, from the vLLM lane's cache. A run needs
three pieces on top of a host-trace torch tree:
1. the overlay on PYTHONPATH, ahead of the site;
2. the traced extension, built against that tree's torch install;
3. ARMV_TRTLLM_CPP=1 in the armV adapter.

## Pieces
- **Patch.** `upstream/src` holds FlashInfer 0.6.18's 6 files: csrc/trtllm_fmha_kernel_launcher.cu and
  include/flashinfer/trtllm/fmha/{fmhaKernels.cuh, fmhaRunner.cuh, fmhaRunnerParams.h, kernelParams.h, lse.cuh}.
  `ht/make_patch.py` writes `src/` from them. Each change is an asserted substitution, and the script fails if a
  substitution no longer matches or if a braced container list remains (the nvcc bug in STATUS.md).
  - `src/` is compiled only into the traced extension, never into FlashInfer's module.
  - `src/UPSTREAM.sha256` holds the upstream hashes. `tests/test_drift.py` checks them against the installed
    FlashInfer.
- **Extension.** `build/fi_ht_trtllm.so` comes from:
  - `ht/ht_traced.cu`: the stock headers, then `src/` inside namespace fi_ht_traced with FI_HT_TRACED. The aliases
    are c10::SymInt and the launches are recorded.
  - `ht/ht_bind.cpp`: the pybind module.
  - Headers `ht/fi_ht.h`, `ht_types.h`, `ht_trace.h`, `ht_entry.h`, and `kernel_params_fields.h` (generated from
    DWARF by `ht/gen_fields.py`).
  - The flags are fmha_gen's (`ht/build_flags.py`), with C++20 for torch's headers. It links against the torch
    install named by `HOSTTRACE_INSTALL`, plus libtvm_ffi.
- **Overlay.** `overlay/flashinfer` holds symlinks to the stock site (land/scratch/sglang/site/flashinfer, read only)
  plus three files:
  - decode.py and prefill.py: a 2-line hook at each trtllm binding call (`traced(binding, kind)`), as the fork had.
  - trtllm_cpp.py: the glue. It holds the custom ops `flashinfer_ht_cpp::trtllm_paged_attention_{decode,context}`,
    turns records into KernelLaunch objects, preloads cubins at the warm-up, runs the byte-for-byte check against the
    stock binding (on by default; FI_HT_CPP_CHECK=0 turns it off), and finds the stock module through
    /proc/self/maps.
  - `jit/` resolves to the site, so FlashInfer's JIT sees the stock sources and reuses the cached fmha_gen.
- **Adapter.** `armV/` is a copy of the attn lane's armV (snap/s5) with ARMV_TRTLLM_CPP=1. That flag behaves like
  the fork's flag (no trtllm_shim, no closed-op harvest) but imports flashinfer.trtllm_cpp. Its summary also gains
  `trtllm_cpp` and `redispatch_ops` (each redispatch's failing op guards).

## Build (CPU only, about 2 min for ht_traced.cu, which is most of it)
```
cd land/core/trtllm_cpp
python3 ht/make_patch.py                                         # src/ from upstream/src
python3 ht/gen_fields.py                                         # only if kernelParams.h changed
taskset -c 108-143 bash python_trtllm_cpp.sh ht/build.py         # build/ against the pinned install
taskset -c 108-143 bash python_trtllm_cpp_rd2.sh ht/build.py     # build_rd2/ against redispatch2's install
taskset -c 108-143 bash python_trtllm_cpp.sh tests/test_drift.py # drift, brace audit, traced shape lists
```
- For another torch tree, set `HOSTTRACE_INSTALL=<its install>` and `FI_HT_BUILD=<a directory>` and run
  `ht/build.py`. Then run with `FI_HT_BUILD` pointing at that directory.
- The extension must be built against the same install the run uses, because it links that install's libc10 and
  libtorch.
- The .so is linked to a temporary file and renamed, so a running process sees either the old file or the new one.

## Run
- Launchers:
  - `python_trtllm_cpp.sh`: the attn lane's launcher on its snapshot s5 (pinned install), with the overlay through
    the ATTN_FORK_DIR slot.
  - `python_trtllm_cpp_rd2.sh`: redispatch2's c6_e on build_rd2/install, with FI_HT_BUILD=build_rd2.
  - Both use FLASHINFER_WORKSPACE_BASE=land/scratch/vllm/cache (the stock modules) and FLASHINFER_NO_DOWNLOAD=1.
- V-stock flags (`vstock.env`): `ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0
  ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0`, INTEG_ARMV=land/core/trtllm_cpp/armV.
- Example: `./gpu1.sh vs_full $(cat vstock.env) -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8
  --mixed-spec 32x512 64x64 --out out/vs_full.json`. Other wrappers: gpu0.sh (short holds) and anygpu.sh (whichever
  lock frees first). Each takes LAUNCHER=.
- Another lane's adapter needs:
  - the overlay ahead of the site;
  - FI_HT_BUILD;
  - no trtllm_shim install (no closed-op harvest of the trtllm ops);
  - `import flashinfer.trtllm_cpp`.
- Cubins:
  - They come from FLASHINFER_CUBIN_DIR (default: the vLLM lane's cache).
  - The warm-up preloads every cubin of the call's class that is on disk.
  - A cubin still needed under a capture declines with "kernel X was not loaded at the warm-up (a capture holds)".
  - bf16 head 256: the vLLM lane fetches it tonight (user-approved) into land/scratch/vllm/cache. The parity family
    bf16_h256_p32_q35_9b runs once those files are there and skips until then.

## What it covers and what it does not
- **Covered:** decode and context, paged KV, every dtype, head and page the launcher selects a cubin for. Verified:
  - bf16 h64/128 p16/64;
  - fp8 h256 p32 at 4, 6 and 8 q heads per KV head;
  - sinks and LSE.
- **Declines, by name:**
  - generation at max_q > 1 (the spec-decode fp32 cost model; waits on the symbolic-float lane's fp32 rows, consumed
    through c10::SymFloat);
  - the separate reduction kernel (GmemReductionWithSeparateKernel);
  - ragged attention and sparse MLA (not compiled in the traced build).

## md5 (2026-10-08 21:40)
make_patch output (`src/`):
```
37cef29bea9d14b599fc60a6f809b0cd  UPSTREAM.sha256
3601d23989813af9197ddb8daba93aec  csrc/trtllm_fmha_kernel_launcher.cu
177959a9d6649afd012592bad0d44095  include/flashinfer/trtllm/fmha/fmhaKernels.cuh
ba30cf191c182cce5b36bc1b63c68e70  include/flashinfer/trtllm/fmha/fmhaReduction.h   (unchanged from upstream)
5e19c4c37af97e49ead6f37a1d7e372e  include/flashinfer/trtllm/fmha/fmhaRunner.cuh
d010b40dee3c9951fcfda9671e2a2aa4  include/flashinfer/trtllm/fmha/fmhaRunnerParams.h
c22c6455af0b5232b2343b6993ee9b21  include/flashinfer/trtllm/fmha/kernelParams.h
d1796be4eb966401c0b7d83d7f76985a  include/flashinfer/trtllm/fmha/lse.cuh
```
Extension source and glue:
```
ffc7a0ceabeae4872b45865099930429  ht/fi_ht.h
bc3ba89c3224aa91aa784aa12ea7f4db  ht/ht_types.h
74026f376d3eca45aca3ca89ebf0b5e4  ht/ht_trace.h
0f8a0507f277f9fbcdd466e0cbe0cc8f  ht/ht_entry.h
e00cb588a235624bf26f233d10bb077c  ht/ht_traced.cu
6244b2325297d137cf45edb3e6384fbc  ht/ht_bind.cpp
c0a3729f466f57dbab16ef70466f0de4  ht/kernel_params_fields.h
171116834f6d5bedac76c964ed13bba1  ht/make_patch.py
b770a3a2d4e908f31aded7a18f2ede9d  ht/gen_fields.py
fe527575a68693da23d1c73e404fc922  ht/build.py
df4f1c6e4da11bf346666678ba2d2771  ht/build_flags.py
638ec7357d4606f8e0323c8fdb417a62  overlay/flashinfer/trtllm_cpp.py
02eba4820ea937d990c05d569cbf0316  overlay/flashinfer/decode.py
7e45100ef81d8d35c403d72cab07ea01  overlay/flashinfer/prefill.py
37f69a41bc452d36f1738af5bcf69e91  armV/adapter.py
409674f69bbcc8cf838b1cc6b9a84da4  build/fi_ht_trtllm.so       (pinned c8x install)
f992c76ea0d14b539bd54fdd86176c12  build_rd2/fi_ht_trtllm.so   (redispatch2 build_rd2 install)
```
These are the files behind the 21:12 to 21:18 results in STATUS.md:
- V-stock 188/188 on both trees;
- parity 219 pass, 0 mismatches;
- trace tests 6/6;
- drift tests 5/5.
