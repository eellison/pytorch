cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f timing_gpu0_e.sh > /dev/null; do sleep 30; done
bash timing_gpu0_s.sh
