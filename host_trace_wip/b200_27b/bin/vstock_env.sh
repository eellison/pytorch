# The V-stock configuration (host-trace replay around stock vLLM, enforce_eager), SOUND: only cuBLAS is harvested; every open op
# (vLLM _C ops through the vllm_ports launchers, Triton, FlashInfer JIT, CuTe DSL, ATen) is traced or runs as an eager step.
# The trtllm-gen attention is traced through FlashInfer's own C++ launcher (the overlay + $FI_HT_BUILD).
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1
