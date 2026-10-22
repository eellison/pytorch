#!/usr/bin/env bash
# mkpatches.sh: the candidate-4 symti patch set in patches/:
#   symti_cpp.patch  C++/torchgen over int6/build_cpp10/src (without torch/csrc/cuda/Module.cpp's jiterator fix)
#   jiterator_null_kwargs.patch  that one-line fix alone (upstream candidate)
#   symti_py.patch   Python + tests over int6/snap/step242 + core/hintaudit/hint_audit.int6cpp.patch + the fuzzfix lane's F20 patch
#   MD5SUMS, READY (written by hand after the gates)
set -eu
L=/data/eellison/src/pytorch/agent_space/paramgraph/land; S=$L/core/symti; P=$S/patches; T=$S/tmp/mkpatches
rm -rf $T; mkdir -p $T
cd $S/src
files=$(diff -rq . $L/int6/build_cpp10/src -x third_party -x .git -x "*.pyc" -x __pycache__ -x "*.egg-info" -x compile_commands.json -x "__init__.pyi" -x generated -x build -x "*.orig" 2>/dev/null \
  | sed -n 's/^Files \.\/\(.*\) and .* differ$/\1/p; s/^Only in \.\(\/[^:]*\)\?: \(.*\)$/\1\/\2/p; s#^Only in '"$L"'/int6/build_cpp10/src\(/[^:]*\)\?: \(.*\)$#GONE \1/\2#p' | sed 's#^/##; s#^GONE /#GONE #')
: > $P/symti_cpp.patch
# a file symti removed (only in build_cpp8, listed as "GONE <path>") diffs to /dev/null
for f in $files; do
  [ "$f" = GONE ] && continue
  case $f in torch/csrc/cuda/Module.cpp) continue;; esac
  a=$L/int6/build_cpp10/src/$f; b=$S/src/$f
  if [ -d "$b" ] || [ -d "$a" ]; then continue; fi
  [ -e "$a" ] || a=/dev/null
  [ -e "$b" ] || b=/dev/null
  diff -uN --label "a/$f" --label "b/$f" "$a" "$b" >> $P/symti_cpp.patch || true
done
diff -uN --label a/torch/csrc/cuda/Module.cpp --label b/torch/csrc/cuda/Module.cpp $L/int6/build_cpp10/src/torch/csrc/cuda/Module.cpp $S/src/torch/csrc/cuda/Module.cpp > $P/jiterator_null_kwargs.patch || true
# Python base: step242 + the hint-audit patch
mkdir -p $T/base
cd $L/int6/snap/step242 && tar cf - torch/cuda torch/_meta_registrations.py test/test_cuda_host_trace_*.py torch/_C/__init__.pyi.in 2>/dev/null | (cd $T/base && tar xf -)
cd $T/base && patch -p1 -s < $L/core/hintaudit/hint_audit.int6cpp.patch && patch -p1 -s < $P/F20_eager_step_output_is_its_argument.symti_py.patch
: > $P/symti_py.patch
cd $S/py
for f in $(ls torch/cuda/_host_trace*.py torch/_meta_registrations.py test/test_cuda_host_trace_*.py); do
  a=$T/base/$f; [ -e "$a" ] || a=/dev/null
  diff -uN --label "a/$f" --label "b/$f" "$a" "$f" >> $P/symti_py.patch || true
done
cd $P && md5sum symti_cpp.patch symti_py.patch jiterator_null_kwargs.patch > MD5SUMS
grep -c '^--- a/' symti_cpp.patch symti_py.patch jiterator_null_kwargs.patch
cat MD5SUMS
