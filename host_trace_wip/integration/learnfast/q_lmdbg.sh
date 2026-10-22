#!/usr/bin/env bash
cd $(dirname $0)
while ! grep -q "^done" logs/mbf.log 2>/dev/null; do sleep 30; done
MIMO_MOE_BUILD=build_fix PROV_DUMP_DIFF=1 BS="1 2" TAG=lmdbg ./q_lift1.sh
