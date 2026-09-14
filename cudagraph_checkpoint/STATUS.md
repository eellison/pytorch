AI-assisted checkpoint note, published at Elias Ellison's request.

We have a validated prototype checkpoint for compiler-driven CUDA graph replay, with both frontend alternatives preserved: FX tracing of generated host code and retained compiler IR. Both feed the same native C++ runtime. Qualified native hits skip generated host Python, while compiler-derived layouts, arguments, output positions, and ownership remain checked during preparation.

The prototype also carries saved-activation release schedules into the native runtime and eager CUDA caching allocator. Supported cases include static and symbolic allocations, exact Tensor/None backward output slots, and release-disabled and retained-graph controls. Tests observe eligible storage reuse in the qualified release workloads; this is not a peak-memory or performance claim.

Two separate qualifications are accepted:

- Original baseline: **252 CPU tests and 44 GPU tests**.
- Latest local-cache extension: **280 CPU tests and 11 GPU tests**, with root and independent evidence reviews accepted. These counts describe separate suites and must not be added together.

The latest tests exercise a real local FX cache miss/save followed by hit/load through both readers, with distinct artifacts and native entries, correct dynamic gradients, and outputs surviving teardown. Support is limited to synchronous local FX caching; remote FX and all AOT cache modes remain excluded. Full native forward/backward execution is not established by these backward tests.

Next: broaden representative workloads and turn the isolated prototypes into a smaller, reviewable integration suitable for landing. Source snapshots, native build identities, successful results, and failed attempts are preserved.
