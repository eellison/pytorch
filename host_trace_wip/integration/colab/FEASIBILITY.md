> **Superseded (2026-10-16).** At the user's direction, the Colab notebooks no longer use the host-trace build. Every
> cell that needed it was removed, along with the setup for it (wheel placeholder, source-build cell,
> `%%needs_host_trace`). They now run only on stock PyTorch; see README_FOR_REVIEW.md. The branch and build analysis
> below is kept for reference, in case a host-trace Colab notebook is wanted later.

# Running the host-trace notebooks in Colab: feasibility (2026-10-15)

Colab today: x86_64, Python 3.12 (3.11 on older images), a CUDA 12.x driver and toolkit. GPUs: T4 (free); L4, A100, H100 (paid).
No Blackwell GPU.

## 1. Which public branch

**`host-trace-wip-20261012-c5`** on https://github.com/eellison/pytorch (checked with `git ls-remote fork`, read-only).

| commit | what |
|---|---|
| `5f756a09c145e0de9f3e202dbf18b310e1c880d2` | the host-trace code: int6 snap/step310 (Python, tests) + build_cpp10/src (C++), unchanged. Base: upstream `350325a750b` (2026-09-17). |
| `ad1477dde66`, `1b3eb8595437` (tip) | only add files under `host_trace_wip/` (B200 bundle); not part of the build. |

Checked: every `torch/cuda/_host_trace*.py` on the branch equals int6 snap/step310, and every branch file under
`torch/csrc/cuda/host_trace` and `aten/src/ATen/native/cuda` equals build_cpp10/src. So step310 on build_cpp10/install
runs the branch's code. Both Colab notebooks' saved outputs come from that combination (GB300, GPU 3).

Other public candidates and why not:
- `host-trace-wip-20261012` (`52a432a8075`, candidate 4, step294) and `host-trace-wip-20261009`: older; no FlashInfer
  mm_fp4 / DataPointer support, no RD2-RD5 redispatch.
- `hosttrace/candidate` = `hosttrace/eager-aten-wip` (`a53e4c2ed42`, int5c, 2026-10-04): much older.
- `host-trace-wip-20261012-b200`: candidate 4 + B200 runbooks.

Not on any public branch: candidate 5 (step328, build_cpp13), which the published CuTe notebook (v3) ran on. Re-running
the CuTe notebook on the branch gave the same outputs except one line: step328's passthrough of loaded functions outside
a trace (`outside a trace, the loaded function is a TVM-FFI Function with its own call`) prints `False` on the branch.
The Colab version drops that line and the sentence claiming it. The Triton notebook (published from candidate 3, step246)
changed more on the branch: the redispatch refusal of a symbolic bool is gone, so plain `user_code.add` now gets 2 traces
and 5 redispatches (was 7 traces, 0 redispatches). Section 6 now runs `user_code.add` only, and its text says so. The
`INT64=False` copy was only there to work around that refusal, so it is dropped.

The branch's commit message says it was pushed before every gate finished, and that one python_entry test is known to fail.

## 2. Feasibility

### (a) Stock nightly plus Python files, or the C++ changes?

- **The toy sections need no host-trace code** (Triton sections 1-5, CuTe sections 1-6). They need PyTorch >= 2.12
  (`torch.cuda._utils._check_cuda_bindings` first shipped in v2.12.0; `CUDAGraph(keep_graph=True)` and `raw_cuda_graph_exec`
  are older), the Triton that PyTorch pins, `cuda-python`, and for CuTe `nvidia-cutlass-dsl==4.6.2` and
  `apache-tvm-ffi==0.1.11`. **Verified:** I ran both notebooks on a torch without host-trace code (the default env's
  upstream-main build, 2.15.0a0), and every toy-cell output matched the branch run exactly (Triton 8/8 cells, CuTe 15/15).
- **The "our implementation" sections need the C++ changes** (Triton section 6, CuTe section 7). `HostTraceReplay`
  subclasses `torch._C._HostTraceEntry`. The branch changes 189 files (+57k lines) outside `host_trace_wip/`:
  `torch/csrc/cuda/host_trace` (10 files), `aten/src/ATen/cuda/host_trace` (10), about 60 files in
  `aten/src/ATen/native/cuda`, `c10/cuda` (the caching allocator, driver_api), `CUDAGraph.cpp`, jiterator, and more.
  A stock wheel plus Python files can't work.

### (b) What a Colab user would have to do for the host-trace sections

1. **A prebuilt x86_64 wheel (recommended).** The notebooks have a `HOST_TRACE_WHEEL = ""` placeholder in the setup cell.
   **We can't build it here:** gb300-box (GB300), dev-box (GB300) and the GH200 hosts are all aarch64, and
   cross-compiling PyTorch with CUDA to x86_64 isn't practical. It needs an x86_64 Linux machine with a CUDA 12.x
   toolkit (to match Colab's driver), Python 3.12 (to match Colab), cuDNN headers if wanted, and then e.g.
   `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9;9.0" USE_DISTRIBUTED=0 BUILD_TEST=0 python -m pip wheel . --no-build-isolation --no-deps -w dist`
   (T4/A100/L4/H100). Every extra arch adds to the CUDA compile time. A self-built wheel doesn't declare a Triton
   dependency: pin the Triton version used at build time (3.8 here) next to it. Host the wheel somewhere public
   (e.g. a GitHub release asset on the fork). A GitHub-hosted runner (4 vCPU, 16 GB RAM, 6 h job limit) is likely too
   slow for a CUDA build without a cache. **Nobody has built the branch for x86_64 yet, or with CUDA 12** (it was built
   and tested on aarch64 with CUDA 13.0, `TORCH_CUDA_ARCH_LIST=10.3`). Its CUDA version guards follow upstream's
   (`CUDA_VERSION >= 12040`, `< 12080`), so nothing obviously needs CUDA 13, but that is untested.
2. **Build from source in Colab.** A commented-out cell in each notebook does it (clone the branch, submodules,
   `pip install -v --no-build-isolation .` with `TORCH_CUDA_ARCH_LIST` = the runtime's GPU, `USE_DISTRIBUTED=0 BUILD_TEST=0 USE_MKLDNN=0`).
   - How the time estimate was made: the `.ninja_log` of a comparable single-arch CUDA build on this box sums to about
     19 CPU-hours of compile on Grace cores (aten ~9 h, torch_cuda ~3.5 h, torch_cpu ~3 h, flash-attention ~2.3 h).
     Colab vCPUs are hyperthreads and slower per vCPU.
   - **Free tier (2 vCPU, ~12.7 GB RAM): not feasible.** That's over 12 hours of wall time, past the session cap, and
     parallel nvcc jobs run out of RAM.
   - **Paid high-RAM runtime (A100/H100, about 8-12 vCPU): roughly 3-5 hours**, plus about 10 minutes for the clone
     and submodules. The session has to stay connected the whole time, and a build is lost with the runtime.
     Possible, but a poor reader experience.
3. **Without either**, the notebooks still run end to end on stock PyTorch. The host-trace cells skip with a note and
   keep their saved outputs, via a `%%needs_host_trace` cell magic defined in the setup cell.

### (c) GPU requirements

| part | GPU | Colab tier | evidence |
|---|---|---|---|
| Triton toy (sections 1-5) | any CUDA GPU Triton supports (sm_75+); needs a CUDA 12.4+ driver for `cuFuncGetParamInfo` | free T4 is enough | run on GB300 only |
| CuTe toy (sections 1-6) | sm_80+ (nvidia-cutlass-dsl 4.6.2's `Arch` enum starts at sm_80: Ampere, Ada, Hopper, Blackwell); **not T4** | paid: L4, A100 or H100. H100 is recommended, the closest to what was tested | run on GB300; compile-only check for sm_80, sm_89 and sm_90a passed (`archcheck/compile_arch.py`, CUTE_DSL_ARCH override; not executed on those GPUs) |
| Triton section 6 / CuTe section 7 | host-trace build; run here on GB300 (sm_103). The host-trace work was also exercised on GH200 (sm_90); never on Ampere/Ada/Turing | paid A100/H100 to build; any GPU the wheel targets | GB300 run |
| CuTe section 7, FlashInfer mm_fp4 cell | Blackwell (sm_100+), FlashInfer 0.6.18 (`mm_fp4(backend="cute-dsl")`) | **not available on Colab** | GB300 run; the cell is tagged `%%needs_host_trace blackwell` and keeps its saved output |

So the task's premise "CuTe DSL needs sm90+" is stricter than what the DSL declares (sm_80+). I kept H100 as the
recommended runtime and marked the notebook that way (Colab metadata `gpuType: H100`), but said L4/A100 should work.

## Untested in Colab (nobody here can run Colab)

- Colab's preinstalled torch/Triton versions. The Triton toy reads Triton internals (`compiled.src.signature`,
  `JITFunction.params[*].annotation_type`, `do_not_specialize_on_alignment`), written against Triton 3.8. Another Triton
  may break `triton_hook.specialization`.
- `cuda-python>=12.6,<13` with Colab's driver.
- Whether `pip install --upgrade torch>=2.12` needs a session restart on a given Colab image (the setup cell says to
  restart if torch was already imported).
