# CUDA graph runtime checkpoint - 2026-09-14

This branch preserves the CUDA graph runtime prototype, its test evidence, and its roadmap.
Prepared with AI assistance at Elias Ellison's request.

See [STATUS.md](STATUS.md) for the milestone summary.

The source archive preserves original file bytes, including the selected compiler/frontends,
native C++ sources, tests, build recipes, and evidence. The manifest records original paths
and SHA-256 hashes. Built libraries, installed dependencies, and duplicate source snapshots
are excluded.

## Restore

```sh
sha256sum -c SHA256SUMS
mkdir restored
tar -xzf sources.tar.gz -C restored
```

Archive paths under `repo/` are relative to the PyTorch checkout.
The original base commit is `d98f9246473655c8b440b2a9612186a4908de4c1`.
The manifest identifies any saved files from outside that checkout.

The saved qualification manifests retain their original absolute paths and native binary
identities. Restoring on another machine requires rebuilding the native extension and
adjusting the bootstrap paths; this archive preserves source and evidence, not an installed
environment.

Latest focused qualification: 280 CPU and 11 CUDA tests passed.
Earlier broader qualification: 252 CPU and 44 CUDA tests passed.
These suites overlap. See the archived results and exact manifests for scope.
