#!/usr/bin/env bash
# Item 3: full range sweeps (rs5 settings) with --diag: before (defaults), ports only, fork only, after (ports + fork).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
B="ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1"
bash bench/run_items.sh rs_before cpp $B -- --diag > bench/logs/range_rs_before.log 2>&1
bash bench/run_items.sh rs_after fork $B ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1 -- --diag > bench/logs/range_rs_after.log 2>&1
bash bench/run_items.sh rs_impls cpp $B ARMV_TRACED_IMPLS=1 -- --diag > bench/logs/range_rs_impls.log 2>&1
bash bench/run_items.sh rs_fork fork $B ARMV_TRTLLM_FORK=1 -- --diag > bench/logs/range_rs_fork.log 2>&1
echo RSDONE > bench/logs/items_rs.done
