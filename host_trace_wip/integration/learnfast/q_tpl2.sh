#!/usr/bin/env bash
# templates without the equality assertion (timing of redispatch), 27B prefill part a, Triton GDN, --check
cd $(dirname $0)
DIAG27_TPL=1 DIAG27_FLA_SYMLEN=1 TAG=${TAG:-l27mf} PART=${PART:-a} MODE=chk GDN=triton G=3 CPUS=108-143 ./q_lift27.sh
