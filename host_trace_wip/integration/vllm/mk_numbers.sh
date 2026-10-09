#!/usr/bin/env bash
cd $(dirname $0)
{ cat NUMBERS_head.md; python3 c2tables.py; [ -f NUMBERS_notes.md ] && cat NUMBERS_notes.md; } > NUMBERS_cand2.md
echo "wrote NUMBERS_cand2.md ($(wc -l < NUMBERS_cand2.md) lines)"
