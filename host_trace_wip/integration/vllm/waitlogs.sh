#!/usr/bin/env bash
# Blocks until one of the given logs changes (or ~110 min), then prints their last lines.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm
s0=$(cat "$@" 2>/dev/null | md5sum)
for i in $(seq 1 110); do [ "$(cat "$@" 2>/dev/null | md5sum)" != "$s0" ] && break; sleep 60; done
date +%T; for f in "$@"; do echo "$f: $(tail -1 $f 2>/dev/null)"; done
