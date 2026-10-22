#!/usr/bin/env bash
# templates keyed by closure roots on 27B (Delta 1 + fla overlay + tpl_closure), Triton GDN, --check part a, with the equality
# assertion (verify_redo_templates: every instantiation also runs the full redo and asserts one entry)
cd $(dirname $0)
DIAG27_TPL=1 DIAG27_FLA_SYMLEN=1 EXTRA_SET=verify_redo_templates=1 TAG=${TAG:-l27ma} PART=a MODE=chk GDN=triton G=3 CPUS=108-143 ./q_lift27.sh
