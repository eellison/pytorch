#!/usr/bin/env bash
# stock3.sh SET [extra llm-kw JSON]: three fresh stock vLLM 0.29 eager processes of Flash-Next NVFP4 (TP1, vLLM's default kernels,
# language model only, max len 16384), each with the quality check; one gpu3.lock hold per process. Run 1 may fetch missing
# FlashInfer cubins (FN_ALLOW_DOWNLOAD=1, new files listed in out/fetched_SET_1.txt). Run under setsid nohup.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
SET=$1; EXTRA=${2:-}
M=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
KW='{"language_model_only": true'"${EXTRA:+, $EXTRA}"'}'
for i in 1 2 3; do
  dl=0; [ $i = 1 ] && dl=${FN_DL1:-1}
  find -L cubins -type f | sort > out/cubins_before_${SET}_$i.txt 2>/dev/null
  FN_ALLOW_DOWNLOAD=$dl INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.88} CMD_TIMEOUT=1200 bash gpu3.sh ${SET}_$i bash python_fn.sh \
    scripts/stock_ref.py --model $M --out out/${SET}_$i.json --llm-kw "$KW" \
    --quality data/ppl_passage_pytorch_README.md data/ppl_passage_lane_status_20261013.md ${FN_STOCK_ARGS:-}
  rc=$?
  find -L cubins -type f | sort > out/cubins_after_${SET}_$i.txt 2>/dev/null
  comm -13 out/cubins_before_${SET}_$i.txt out/cubins_after_${SET}_$i.txt > out/fetched_${SET}_$i.txt
  echo "$(date +%T) ${SET}_$i rc=$rc fetched $(wc -l < out/fetched_${SET}_$i.txt)"
done
python3 scripts/cmp_refs.py out/${SET}_1.json out/${SET}_2.json out/${SET}_3.json
