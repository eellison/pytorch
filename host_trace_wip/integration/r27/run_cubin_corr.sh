#!/usr/bin/env bash
# Does a process's GDN NaN set follow its compiled kernel? gdn_rep2.py (stock) in 6 fresh processes with the CuTe DSL keeping PTX and
# cubins (CUTE_DSL_KEEP=ptx,cubin) in a per-process dump dir; then cubin / PTX hashes per kernel vs the process's NaN shapes.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
for k in 1 2 3 4 5 6; do
  d=$R/out/ficheck/cubins/p$k; rm -rf $d; mkdir -p $d
  echo "proc $k: $(CUTE_DSL_KEEP=ptx,cubin CUTE_DSL_DUMP_DIR=$d timeout 600 bash $R/python_r27.sh $R/scripts/gdn_rep2.py 2>&1 | grep -E '^1x|^4x' | awk '{split($4,a,"/"); printf "%s %s ", $1, (a[1]=="0" ? "ok" : "NaN")}')"
  echo "   files: $(find $d -type f | wc -l); cubin md5: $(find $d -name '*.cubin' -exec md5sum {} + | awk '{print substr($1,1,8)}' | sort | tr '\n' ' ')"
  echo "   ptx md5: $(find $d -name '*.ptx' -exec md5sum {} + | awk '{print substr($1,1,8)}' | sort | tr '\n' ' ')"
done
