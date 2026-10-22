#!/usr/bin/env bash
# Blocks until the number of lines matching REGEX across FILES grows (or ~115 min): waitpat.sh REGEX FILES...
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm
re=$1; shift
n0=$(cat "$@" 2>/dev/null | grep -cE "$re")
for i in $(seq 1 115); do [ "$(cat "$@" 2>/dev/null | grep -cE "$re")" != "$n0" ] && break; sleep 60; done
date +%T; cat "$@" 2>/dev/null | grep -E "$re" | tail -5
