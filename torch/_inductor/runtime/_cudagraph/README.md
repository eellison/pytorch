# Parameterized CUDA graph runtime

This experimental runtime traces host execution after ordinary kernel selection,
then replays the resulting CUDA graph with updated addresses, integer parameters,
and launch dimensions. The Python modules prepare the replay; the native entry
evaluates reuse guards, allocates storage, patches the graph, submits it, and
constructs returned tensors.

`api.py` provides the direct-host entry and the Inductor policy. Direct hosts use
an explicit input contract and invocation adapters. Inductor supplies the input
facts it already knows. A reuse-guard miss runs the ordinary host, traces the
selected execution, and prepares another variant.

## Source map

| Component | Main files |
| --- | --- |
| Direct Python host and invocation observation | `direct_host.py`, `direct_triton.py`, `direct_cute.py` |
| Triton tensor maps and launcher scratch | `triton_tma.py`, `triton_scratch.py` |
| Inductor attachment and policy | `metadata.py`, `policy.py` |
| Symbolic host tracing and storage addresses | `extraction.py`, `address_trace.py`, `trace_views.py` |
| Shared lowering and reuse guards | `frontend.py`, `guard_export.py`, `address_scalars.py` |
| CuTe conversion, host IR, and physical parameters | `cute_adapter.py`, `_compiler/cute_dispatch/`, `_compiler/argument_flow.py`, `_compiler/cute_bridge/` |
| Capture and native replay preparation | `replay.py`, `../cudagraph_boxed_replay.py` |
| Native boxed entry and typed parameter evaluation | `torch/csrc/cuda/Graph.cpp`, `torch/csrc/cuda/GraphParameterProgram.h` |
| Graph parameter access and updates | `aten/src/ATen/cuda/CUDAGraphParams.cpp` |

The last three native paths are relative to the repository root. The `_compiler`
package contains the shared host-analysis helpers used during preparation.

## Contracts

- Trace applicable dynamic sizes and strides, and symbolize storage addresses.
  Preserve known static input facts. Views compose with their existing storage
  root and offset expressions.
- An `InputSource` address is the input tensor's `data_ptr()`, including its
  storage offset. It is not the allocation base. Empty tensors have null data
  pointers; a nonempty view cannot currently be rooted in an empty input.
- Record allocations, views, invocations, and outputs in host order. Kernel
  mutation remains kernel work; host tracing does not functionalize the program.
- Use the actual selected Triton launcher metadata or CuTe host-IR dataflow to
  map symbolic operands to physical kernel parameters. CUDA layout queries
  check offsets and widths. Correspondence is not inferred by fitting values.
- Treat kernel bodies as opaque. Every passed storage address constitutes a use.
  CuTe IR interpretation here concerns the host computation and launch ABI.
- Keep reuse guards local to a prepared variant. Preserve selected alignment,
  specialization, integer-width, and arithmetic-domain requirements. These are
  distinct from the caller's compilation guards.
- Replay owns its required modules and storage. Returned tensors can outlive the
  runtime. Saved-backward-input reclamation is optional and uses last-use data.

Triton host descriptors lower to a whole 128-byte tensor-map parameter followed
by the selected ABI's shape and stride scalars. `_TensorMapField` carries its
storage root, byte displacement, dimensions and byte strides in CUDA axis order,
plus compiler-selected encoding constants. Preparation converts the integer
expressions into early numeric-program indices in a typed native binding. Both
capture and replay use CUDA's tiled tensor-map encoder; capture checks the full
argument bytes and the loaded function's parameter widths. Descriptor roots
participate in the same lifetime tracking as ordinary pointer arguments.

Device-created Triton descriptors use launcher scratch. Each nonempty compiler
scratch slot receives an owned allocation sized by its per-CTA requirement times
the symbolic grid volume; empty trailing slots remain null. Scratch alignment
must fit the eager allocator guarantee. The replay owns scratch through graph
submission and borrows the selected kernel module until close.

The Inductor policy is an identity-bearing runtime owner. Configuration snapshots
share that owner instead of duplicating its locks, installations or CUDA graphs.
Its ordinary callable can be the generated function or Inductor's explicit input
alignment wrapper. Native variants trace the generated function and guard that
the wrapper would make no copies. Warmup and misses call the ordinary wrapper,
preserving its existing copy and mutation-writeback behavior. Closing the
installation restores that same ordinary callable.

## Development and tests

The component suites and commands are in
`test/inductor/cudagraph_runtime/README.md`; native graph API tests are in
`test/test_cuda.py`. CPU scripts run through pytest-xdist, and CUDA scripts run
serially in isolated processes. Larger compositions supplement the component
tests for ABI, guards, arithmetic, views, selection, and lifetime behavior.

CuTe currently requires version 4.6.2 and activation of `_sdk.activate()` before
importing CuTe or constructing runtime entries:

```python
from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass.cute as cute
```

Activation provides raw MLIR values during host inspection and installs the
compiled-call hooks before conversion entries snapshot their dependencies.
Installing hooks after creating an entry changes those dependencies and
invalidates it. Activation does not replace the installed SDK. `ObservedOrdinaryEntry` checks activation
before preparing either a direct or Inductor CuTe invocation and reports the
required import order. Triton-only direct execution does not need CuTe. The
private compiler interfaces and unsupported launch forms remain prototype
integration constraints.

Current qualification uses Python 3.12, CUDA 13, Triton 3.8.0+gitb252c7c4, and
cuda-bindings 13.3.1 on GB300. CuTe TMA tests require Blackwell; Triton TMA tests
require SM90 or newer and the corresponding Triton descriptor APIs. These results do
not establish coverage across other compiler versions or GPU architectures.
