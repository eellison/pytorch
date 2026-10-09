#!/usr/bin/env bash
# core_gaps_repro.py on the pinned build and on candidate 3
CAND_LINE=pinned bash python_vllm_cand2.sh serve/repro/core_gaps_repro.py; echo "pinned rc=$?"
bash serve/python_vllm_cand3.sh serve/repro/core_gaps_repro.py; echo "cand3 rc=$?"
