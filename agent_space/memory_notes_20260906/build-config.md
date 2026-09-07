---
name: build-config
description: "How to build PyTorch in this repo (env vars, conda env, ccache, runtime LD_LIBRARY_PATH)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 61d2d8b8-c61d-4f4f-9055-7ebb3d7562a6
---

Build command (per CLAUDE.md, the ONLY build command): `pip install -e . -v --no-build-isolation`.

Environment for building (B200 box, conda env `pytorch-3.12`):
- `TORCH_CUDA_ARCH_LIST=10.0` (B200 / sm_100) to avoid building all archs.
- ccache at `/home/eellison/local/ccache/bin/ccache` is on PATH and warm; set `CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache`.
- Python/pip is `/home/eellison/.conda/envs/pytorch-3.12/bin/python` (CONDA_PREFIX=/home/eellison/.conda/envs/pytorch-3.12).
- A full rebuild from a large jump takes a few minutes with warm ccache.

Runtime: every `python` invocation needs `LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib:${LD_LIBRARY_PATH:-}` (otherwise `import torch` fails with `GLIBCXX_3.4.30 not found`). Also set `PYTHONPATH=/data/users/eellison/pytorch`.

Two environment traps that each cost a failed configure (hit 2026-09-03):
- `CCACHE_DIR` in the environment points at `/data/eellison/ccache`, which does
  not exist (note the missing `users/`). ccache then dies with
  `Failed to create directory .../lock: Permission denied` and CMake reports the
  confusing `Couldn't determine version from header`. The real warm cache is
  `/home/eellison/.cache/ccache` (25G). Always `export CCACHE_DIR=/home/eellison/.cache/ccache`.
- `/usr/local/cuda` symlinks to 12.8, but the working build uses **13.0**
  (`CUDA_HOME` is already 13.0). If CMake picks 12.8 you get a stale-cache
  `CMAKE_CUDA_COMPILER` conflict. Pin `CUDA_HOME=CUDAToolkit_ROOT=/usr/local/cuda-13.0`,
  `CUDACXX=/usr/local/cuda-13.0/bin/nvcc`, and put `/usr/local/cuda-13.0/bin` on PATH.

A from-scratch build (no `build/` dir) with the warm cache is ~3539 ninja steps,
about 25 minutes wall clock.

ROCm submodules (third_party/composable_kernel, aiter, flash-attention 3rdparty) may fail to fetch; not needed for the CUDA build. Init CUDA submodules explicitly if needed (third_party/cutlass etc.).
