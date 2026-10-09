#!/usr/bin/env bash
# core_gaps_repro.py (gap_5 nested keyed site in a guarded custom op + the others) on the pinned build
CAND_LINE=pinned bash python_vllm_cand2.sh serve/repro/core_gaps_repro.py; echo "pinned rc=$?"
