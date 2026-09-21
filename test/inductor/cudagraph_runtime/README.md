These tests exercise the installed branch runtime through normal imports. The
full catalog needs Triton and the supported CuTe SDK (qualified with 4.6.2).
No scratch bundle, source finder, or personal checkout path is required.

Run both groups through PyTorch's standard runner:

```sh
python test/run_test.py -i inductor/test_cudagraph_runtime_cpu inductor/test_cudagraph_runtime_cuda
```

The standard runner requires PyTorch's usual CI pytest dependencies, including
`pytest-rerunfailures`, from `.ci/docker/requirements-ci.txt`.

The standard entries launch each catalog script in a fresh interpreter, preserving
its local fixture imports and `common_utils.run_tests` behavior. Discovery excludes
the nested scripts and xdist coordinator to avoid duplicate runs. The CUDA entry
is in `RUN_PARALLEL_BLOCKLIST`, so the standard runner serializes it even when
other tests use parallel workers.

The optional CPU xdist utility and serial CUDA utility use the same launcher:

```sh
CUDA_VISIBLE_DEVICES="" python -m pytest -n 2 -c test/inductor/cudagraph_runtime/parallel_cpu/pytest.ini test/inductor/cudagraph_runtime/parallel_cpu
python test/inductor/cudagraph_runtime/run_cuda.py
```

Missing `cutlass` explicitly skips only scripts marked `requires_cute` in the
catalog. An installed SDK that fails to import or compile still fails the test.
The API identity script keeps its general checks available without CuTe and skips
only its two SDK-dependent identities. The standard CUDA entry skips when CUDA
is unavailable; individual hardware requirements remain in the leaf tests.

Each failed child reports its command path, exit status, stdout and stderr. CPU
xdist workers also retain stdout/stderr files. Individual scripts accept the usual
test-runner arguments, for example:

```sh
python test/inductor/cudagraph_runtime/cpu/test_numeric_program.py -v
python test/inductor/cudagraph_runtime/cuda/generated/test_generated_selection.py -v
```

`suite.json` is authoritative for script inventory and parameterized case counts. Standard-runner reporting has one outer case per script, plus
four CPU launcher checks; those outer cases are not extra component coverage.
The catalog is not a passing-test receipt. [COVERAGE.md](COVERAGE.md) maps the
component contracts. Native graph tests in `test/test_cuda.py` and owned-allocation tests in
`test/test_cuda_graph_owned_alignment.py` run separately. Fresh-thread parameter
update tests are in `test/inductor/test_cudagraph_host_trace_threads.py`.

The `cuda/test_hosttrace_*.py` catalog entries exercise the canonical CUDA host
adapter, including pointer tables, pinned-input copies, reductions and RNG.
Raw recorder,
authority, flash, TensorIterator and reduction tests are in
`test/test_cuda_host_trace*.py`. Shared replay and frontend contracts are in
`test/inductor/test_cudagraph_host_trace*.py`; these include mapping, guards,
lowering, metadata, payload simplification, flash and TensorIterator. These tests
outside the catalog are not included in its component counts. CUDA bridge tests require
NVIDIA CUDA 12.8 or later and the corresponding host-tracing native bindings.

`inductor/test_cudagraph_compiled_registration` tests the native evaluator ABI,
execution ordering and library lifetime. The CPU catalog also checks compiled
integer and parameter semantics against the interpreters. Inventory and test
availability do not imply execution coverage; retain actual test-run receipts
when qualifying a source/build pair.

TensorOp and TMA tests load unchanged upstream kernels from this checkout's
`third_party/cutlass` submodule. Initialize it before running them. CuTe TMA requires
SM10x Blackwell and retains its hardware skip condition. Temporary compiler diagnostics
are scoped to each test.
