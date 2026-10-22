#!/usr/bin/env bash
# Vc+VP decode across consecutive batch sizes 4 -> 20 (one variant replayed across sizes), --check: outputs, GDN state and vLLM's sampler
# tokens compared every decode step; then decode bs 1-128 without --check (the n27w sweep's decode set) for the crash
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
IMA_BS="$(seq -s ' ' 4 20)" IMA_CHECK=1 IMA_TAG=vpbs_c12 bash $R/run_ima_c12.sh > $R/logs/vpbs_c12.log 2>&1; grep -E "illegal|rc=" $R/logs/vpbs_c12.log
python3 $R/b200_bundle/tools/verdict.py $R/out/vpbs_c12.json | head -4
IMA_BS="$(seq -s ' ' 1 128)" IMA_TAG=vpbs128_c12 bash $R/run_ima_c12.sh > $R/logs/vpbs128_c12.log 2>&1; grep -E "illegal|rc=" $R/logs/vpbs128_c12.log; grep -c "^decode" $R/logs/vpbs128_c12.log
