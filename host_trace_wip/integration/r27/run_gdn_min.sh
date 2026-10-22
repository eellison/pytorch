cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
for i in 1 2 3; do bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/python_r27.sh -u /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/scripts/gdn_min.py 2>&1 | grep -v Warning; done
