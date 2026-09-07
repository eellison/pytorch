# All-Reduce Template Fusion Notes

Current checkpoint: `9b38cb10070 [inductor] Prototype template nested reduction epilogues`.

## What Works

The checkpoint teaches a Triton template to expose a tile as a reduction-shaped
epilogue:

```python
store_reduction_output(
    ("idx_m",),
    ("idx_n",),
    "tmp",
    mask="mask",
    val_shape=("BLOCK_M", "BLOCK_N"),
)
```

That is enough for a template producing an `[M, D]` tile to fuse:

1. a reduction-shaped epilogue such as RMSNorm, and
2. a nested grouped reduction such as block amax over the normalized output.

The focused tests pass for local template all-reduce stand-ins.

## The Real All-Reduce Gap

A real symmetric-memory all-reduce template needs launch-time communication
state that normal `TritonTemplate` kernels do not currently model.

The older Lamport prototype takes:

```python
buffer_ptrs_dev
counter_ptr
phase_ptr
input_ptr
bias_ptr
weight_ptr
output_ptr
```

Normal template plumbing knows about graph tensor inputs, graph outputs,
sizevars, grid args, and one ordinary workspace. It does not have a first-class
way to add runtime handle fields such as:

```python
symm_mem.rendezvous(symm_buffer, group).buffer_ptrs_dev
symm_mem.rendezvous(symm_buffer, group).signal_pad_ptrs_dev
```

as kernel arguments.

## Why Lamport Is More Than a Template Body

The Lamport version is fast because it avoids barriers by using a rotating
triple buffer and a persistent phase:

1. each rank writes into the current slice of every peer's symmetric buffer;
2. each rank polls its own slice until peer writes have arrived;
3. one CTA advances the phase after all CTAs finish.

That requires state to persist across invocations:

- the symmetric triple buffer must keep the same device address;
- the counter/phase state must be stable across calls;
- stale data must not be mistaken for newly arrived data.

`WorkspaceArg` is not enough for the triple buffer because it is not a
symmetric-memory allocation. `CommBufferLayout` is closer: wrapper allocation
uses `empty_strided_p2p(..., alloc_id=...)`, which is persistent and rendezvous
caches the symmetric-memory handle for the same allocation.

## Reasonable Implementation Path

1. Add a small template runtime-arg mechanism.

   A symmetric-memory template needs to emit a wrapper preamble:

   ```python
   symm_hdl = symm_mem.rendezvous(symm_buffer, group_name)
   ```

   and pass fields such as `symm_hdl.buffer_ptrs_dev` to the Triton kernel.

2. Start with an explicit symm-buffer template op.

   A practical first target is closer to:

   ```python
   symm_mem.one_shot_all_reduce_copy(symm_buffer, local_input, "sum", group)
   ```

   than raw `_c10d_functional.all_reduce`, because the staging buffer is already
   part of the op contract.

3. Model persistent Lamport state deliberately.

   Either require an explicit workspace/state object, or add hidden persistent
   `CommBufferLayout` buffers for the triple buffer and phase/counter state.
   The latter is better for normal all-reduce lowering, but is a larger IR and
   allocation-planning change.

4. Reuse the reduction-epilogue path.

   Once the template produces the all-reduced `[M, D]` tile, the existing
   `store_reduction_output()` path can fuse RMSNorm and the nested amax.

## Constraints

- `@requires_shmem` only accepts a raw `@triton.jit` function. It is not
  directly compatible with Inductor's `@triton_heuristics.template` decorator.
- The Lamport prototype avoids SHMEM device-library calls by taking peer buffer
  pointers directly, but that means the wrapper must pass handle fields.
- The current machine reports `symm_mem.is_nvshmem_available() == False`, so we
  can validate codegen shape locally but not run the real distributed kernel.
