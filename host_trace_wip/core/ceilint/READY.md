# READY: core ceilint (candidate 4)

- `ceilint.int6cpp.patch` (md5 e6e8f95b5ccd9116caf9d8ae4e08a32a): merged in int6 at step271 (odd) / step272 (even).
- `ceilint_floorratio.cand4.patch`, md5 fa434d42dc70da90e17f2908a5affa8f: the same file for the odd top step283
  and the even top step288 (also applies to step290; its five files are identical across 275-290). Python only.
  It contains the IR `floorratio` node for math.floor of an int ratio (with the |n| <= 2**53-1 bound), and these
  oracle changes: strict mode (TORCH_HOST_TRACE_ORACLE=strict / install(strict=True): a difference fails the test),
  LoweredSelector row pairing, guard rows from every Lowering in order, and floorratio / Eq(rel, bool) canonicalization.
  Plus tests.

Superseded: `ceilint_floorratio.int6cpp.patch` (on step246). Details, the oracle census and the difference report
are in STATUS.md, in the "Candidate 4" section.

- Even line on snap/step310: `ceilint_floorratio.step310.int6cpp.patch`, md5 fd5ccf097a4eb1e63d7af62aead3fd83. Applies
  with no offsets or fuzz. Hunk 2 of _host_trace_ir.py is rebased over S1's f32i/f32r `_BUILDERS` entries; the
  content is the same as cand4. CPU checks on build_cpp10: test_cuda_host_trace_lower 24 OK, and the new floor test
  fails on unpatched step310. GPU suites were not rerun: there is no sm_90 build10 tree on gh200-b. The odd line
  is merged at step297. Sent to the int6 agent.
- Merged: even at int6 step312 (on top of step310, 0 offsets, md5 matches); odd at step297. int6 is running the GPU
  gates (lower, replay) on GB300 build_cpp10 and will report any failure.
