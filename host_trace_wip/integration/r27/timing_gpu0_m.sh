#!/usr/bin/env bash
# Mixed-step timing on c5h = "candidate 5 step324, gates pending" (int6 snap/step324 + build_cpp12, build_c12; cute_arith merged): 3 rounds of round_27s_line.sh on GPU 0 (idle-gated,
# gpu0.lock per round), arms default / default-nc / Vc+VP (VP_STOCK), every hybrid kind replayed (R27_EAGER_KINDS empty).
# Shapes (g4-style): one 1024 / 2048-token chunk + 8 / 64 decodes at decoder KV 1024-2048 and 6000-8000; per shape the first two
# mixed steps (trace / warm-up) are dropped by drive_fn, 4 measured. DIAGNOSIS CONFIG: gdn_prefill_backend=triton.
# R27_VC_EXTRA: extra env for the Vc arms (none for c5h).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
MIX="1024x8@1024-2048 2048x8@1024-2048 1024x64@1024-2048 2048x64@1024-2048 1024x8@6000-8000 2048x8@6000-8000 1024x64@6000-8000 2048x64@6000-8000"
W="--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ix-mixed $MIX --ix-reps 6 --llm-kw '{\"kv_cache_dtype\": \"fp8\", \"max_num_seqs\": 128, \"gdn_prefill_backend\": \"triton\"}'"
for r in 1 2 3; do
  echo "$(date +%T) round $r waiting for an empty GPU 0 and gpu0.lock"
  until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  flock -x $E/gpu0.lock env R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12 \
    R27_ARMS="${R27_ARMS:-default fullnc vcvp}" R27_EAGER_KINDS= R27_W="$W" R27_VC_EXTRA="${R27_VC_EXTRA:-}" SET=${SET:-n27m} R=$r HT_GPU=0 R27_KV_GIB=40 \
    taskset -c 0-71 bash $R27/round_27s_line.sh
  echo "$(date +%T) round $r done"
done
python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing ${SET:-n27m} | tee $R27/out/timing/TABLE_${SET:-n27m}.txt
