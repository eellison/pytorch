#!/usr/bin/env bash
# After norejit's range run ends: stop norejit's job and the queued vp_pinned waiter; run c3nan first, then the rest.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "end c2_pinned_norejit_range" vllm/logs/q_norejit_gpu0_c2.log; do sleep 10; done
kill $(pgrep -f "vllm/queue_after.sh queue_gpu0aa.log") 2>/dev/null
for p in $(pgrep -f "vllm/queue_gpu0.sh"); do grep -q norejit /proc/$p/environ 2>/dev/null && kill $p; done
kill $(pgrep -f "vllm/norejit_gpu0_c2.sh") 2>/dev/null
sleep 2
for p in $(pgrep -u eellison -f "vllm/out/t_norejit_r1.json"); do kill $p; done
JOBLIST="c3nan_gpu0_c2;norejit_gpu0_c2;vp_pinned_gpu0_c2" exec bash vllm/queue_gpu0.sh
