#!/usr/bin/env bash
# --check subset, ports + fork on (ck_after) then off (ck_before).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
SUB="--check --dec-bs 1 2 3 8 17 33 64 --pre-lens 2 17 64 512 2048 8192 --pre-multi 2x9 4x128 2x4096 --mixed-nd 1 2 8 --mixed-lens 1 17 128 8192"
bash bench/run_items.sh ck_after fork ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1 ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1 -- $SUB > bench/logs/range_ck_after.log 2>&1
bash bench/run_items.sh ck_before cpp ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1 -- $SUB > bench/logs/range_ck_before.log 2>&1
echo CK1DONE > bench/logs/items_ck1.done
