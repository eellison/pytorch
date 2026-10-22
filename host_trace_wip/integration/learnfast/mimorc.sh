#!/usr/bin/env bash
# mimorc.sh TAG MODE: MiMo-V2.6-Flash Vc under diag27.py with the redo_context prototype (caller's DIAG27_* env), the MiMo
# lane's diag_dcurve config (python_mimo.sh: step331 + build_cpp14, their overlays and fast bindings, read only).
# MODE dc: diag_dcurve's points (1K ctx, bs 1-8 16 24 32, 128 steps); chk: --check (every forward vs eager, twice) on the same
# points with 4 steps each. Writes only into learnfast/out. GPU $LF_GPU (caller holds the lock), CPUs $LF_CPUS.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_moe_cpp
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
LF=$L/core/learnfast
TAG=$1 MODE=$2
cd $L/scratch/integration
export VLLM_USE_DEEP_GEMM=1 MIMO_BACKPORT=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual
export ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1
export ARMV_EAGER_KINDS= ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_INDUCTOR_SEAMS=1 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_REQUIRE_WARM=1
export ARMV_BINDINGS_LOAD=$D/bindings/loctest_vc.fast.pkl INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((40 << 30))
export ARMV_DUMP_VARIANTS=$LF/out/${TAG}_variants.json DIAG27_FILE=$LF/out/$TAG.diag.jsonl CUDA_VISIBLE_DEVICES=${LF_GPU:-3}
export DIAG27_REDOCTX_LAZY=${REDOCTX_LAZY:-0}  # LAZY=1 (static prefix by identity) declines on MiMo: FA4 kv cache / MoE weights reach launchers untracked (mrc_dc)
# with the redo_context prototype the adapter's interim section refusal (RD2 baked sizes) is lifted, as no_sections is in diag27
[ "${DIAG27_REDOCTX:-0}" = 1 ] && export ARMV_REFUSE_SECTIONS=${ARMV_REFUSE_SECTIONS:-0}
BS=${BS:-1 2 3 4 5 6 7 8 16 24 32}
if [ "$MODE" = chk ]; then export ARMV_CHECK_TWICE=1; X="--check --dcurve-steps 4 --dcurve-warm 2"; else X="--dcurve-steps 128 --dcurve-warm 16"; fi
timeout ${RUN_TIMEOUT:-1150} taskset -c ${LF_CPUS:-108-143} bash $D/python_mimo.sh $LF/diag27.py $L/core/r27/bench/drive_sweep27.py --arm Vc \
  --model /data/eellison/models/MiMo-V2.6-Flash-MOPD --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 \
  --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 1024 --dcurve-bs $BS $X \
  --llm-kw '{"compilation_config": {"cudagraph_mode": "NONE"}}' --out $LF/out/$TAG.json
echo "rc=$?"
[ "$MODE" = chk ] && python3 $L/core/r27/b200_bundle/tools/verdict.py $LF/out/$TAG.json
