#!/usr/bin/env bash
# After CHANGES item 22: packed test, xcheck_vllm (fork launcher), then the --check subset with ports + fork on (ck_after2).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
cd $V
LK="flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock timeout 3000 taskset -c 108-143"
CUDA_VISIBLE_DEVICES=2 $LK bash python_vllm_cpp_fork.sh -m pytest -q src_vllmcpp/test/test_cuda_host_trace_packed.py -p no:cacheprovider > bench/logs/t_packed.log 2>&1
echo "packed rc=$?"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=2 $LK bash python_vllm_cpp_fork.sh $S/armF/fork/xcheck_vllm.py > bench/logs/xcheck_vllm_fork.log 2>&1
echo "xcheck rc=$?"
SUB="--check --dec-bs 1 2 3 8 17 33 64 --pre-lens 2 17 64 512 2048 8192 --pre-multi 2x9 4x128 2x4096 --mixed-nd 1 2 8 --mixed-lens 1 17 128 8192"
bash bench/run_items.sh ck_after2 fork ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1 ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1 -- $SUB > bench/logs/range_ck_after2.log 2>&1
echo CK2DONE > bench/logs/items_ck2.done
