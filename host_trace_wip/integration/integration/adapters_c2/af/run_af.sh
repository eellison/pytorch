#!/usr/bin/env bash
# Arm Af correctness and outside-graph counts on GPU 1 (no timing). One gpu1.lock hold per group, bounded by TMO.
# Usage: run_af.sh <group>...; groups:
#   seq     PREP_SEQCHECK (prefill + 13 decode steps, bs 1 8 13 64): E (eager, graphs off), A (stock graphs), Af
#   seqL    the same with PREP_SEQ_LEN=250 (page crossings)
#   profA   one profiled decode step at bs 8 and 64 (whole mr.decode): A, Af
#   profB   the same for F (ARMF_PREP=0), P (ARMF_PREP=1), Pc (PREP_CPP=1, adapters_c2/adapter_prep_cpp.py)
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $S
L=$X/python_sgl_cand2_even.sh; O=$X/out_af; G=$X/logs_af
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
BASE="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 32768 --attention-backend trtllm_mha"
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"
DRV=armF/prep/driveF_prep.py
AF=ARMF_ADAPTER_DIR=$X/adapters_c2/af
one() {  # name, graphs (on|off), env..., -- extra args
  local name=$1 g=$2; shift 2; local env=(); while [ "$1" != "--" ]; do env+=("$1"); shift; done; shift
  local x=""; [ "$g" = off ] && x="$DIS"
  echo "--- $name $(date +%T)"
  env "${env[@]}" bash $L $DRV $BASE $x "$@" > $G/$name.log 2>&1
  echo "rc=$? $name $(date +%T)"
}
group() {
  case $1 in
    seq|seqL)
      local sfx="" sl=(); [ $1 = seqL ] && sfx=L && sl=(PREP_SEQ_LEN=250)
      for a in E A Af; do
        case $a in E) g=off; e=(ARMF_OFF=1);; A) g=on; e=(ARMF_OFF=1);; Af) g=on; e=($AF);; esac
        one seq${sfx}_$a $g "${e[@]}" "${sl[@]}" PREP_SEQCHECK=$O/seq${sfx}_$a.json -- --batch-size 1 8 13 64 --prefill-lens 64 --out /dev/null
      done;;
    profA|profB)
      local arms; [ $1 = profA ] && arms="A Af" || arms="F P Pc"
      for a in $arms; do
        case $a in
          A) g=on; e=(ARMF_OFF=1);; Af) g=on; e=($AF);; F) g=off; e=(ARMF_PREP=0);; P) g=off; e=(ARMF_PREP=1);;
          Pc) g=off; e=(ARMF_PREP=1 PREP_CPP=1 ARMF_ADAPTER_DIR=$X/adapters_c2);;
        esac
        one prof_$a $g "${e[@]}" -- --batch-size 8 64 --prefill-lens 64 --reps 2 --decode-steps 8 --profile-prefill --profile-decode 8 64 \
          --out $O/prof_$a.json --profile-dir $O/prof_$a
      done;;
  esac
}
if [ "$1" = --inner ]; then echo "LOCKED $(date +%T)"; group "$2"; exit; fi
for grp in "$@"; do
  echo "=== $grp queued $(date +%T)"
  CUDA_VISIBLE_DEVICES=1 flock -x $LOCK timeout ${TMO:-1800} taskset -c 72-107 bash $0 --inner $grp
  echo "=== $grp rc=$? $(date +%T)"
done
