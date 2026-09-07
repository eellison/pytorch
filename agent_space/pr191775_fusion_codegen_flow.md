# #191775 fusion-to-codegen flow

This diagram is the high-level map for the exact-dependency prerequisite and
the MXFP6 layer. It intentionally separates fusion-time proof from codegen-time
planning: loop merging can change dependency indices, so the fusion plan is not
cached into codegen.

```text
producer -> consumer
        |
        v
recognize nested/sub-parent topology
        |
        +-- unrelated pair ------------------------> ordinary fusion
        |
        +-- recognizable but no valid plan -------> reject
        |
        v
build StagedReductionPlan
        |
        v
validate raw INTERLEAVED frames
        |
        +-- indirect or ambiguous X | R frame ----> reject
        |
        v
prove residual consumer reads of producer outputs
        |
        +-- exact ordinary dependency -------------> normal match
        +-- exact ProjectedSourceAccess pair ------> planned match
        +-- inherited grouped-stage relation ------> scoped legacy proof
        +-- anything else -------------------------> reject
        |
        v
score reuse + suppress speculative loop rewrites
        |
        v
ordinary or staged vertical legality + backend capability gate
        |
        v
create fused staged node
        |
        v
merge_loops / dependency refresh invalidates fusion-time plan
        |
        v
codegen rebuilds StagedReductionPlan from final fused topology
        |
        +-- plan lost ------------------------------> compiler assertion
        |
        v
planner orders internal-source closure at the end of the parent schedule
        |
        v
generate_node_schedule ensures that closure follows a completed reduction loop
        |
        +-- final loop already open ----------------> reuse it
        +-- reduction loop still active -----------> close it once
        |
        v
emit parent -> grouped reduction -> sub-parent output groups

Append side path:
existing FusedNestedReductions + pointwise consumer
        -> build prospective grouped node and plan
        -> verify every appended node has a planned domain
        -> run the same dependency proof and fusion legality
        -> fuse_with rebuilds the plan before mutating the fused node

existing exact FusedStagedReduction + parent-shaped pointwise consumer
        -> rebuild the complete parent + 4:3 + append plan
        -> require exact projected dependency coverage
        -> run ordinary legality
        -> append, allowing an aligned scale swizzle to remain in one kernel

A shifted scale read fails the dependency proof. Initial formation still
requires the consumer to be wholly in the derived stage; only the already
planned exact staged node receives this append path.
```

## Review checkpoints

1. **Recognition is fail-closed.** A shape that looks like a sub-parent
   candidate but cannot produce a complete proof must reject; it must not fall
   through to ordinary fusion and loop reindexing.
2. **The positive artifact is the plan.** `_can_fuse` carries a non-optional
   `StagedReductionPlan` locally only after branch-local recognition has ruled
   out the fail-closed cases. It then derives the exact `MemoryDepMatch` tuple once.
   An empty tuple still belongs to a real staged plan.
3. **Sub-parent ownership is stricter.** New cross-domain reads require an exact
   plan record. Only reads owned exclusively by the inherited grouped stage may
   use the older nested equivalence proof.
4. **Raw frames stay meaningful.** Before normalized indices are compared,
   multi-axis INTERLEAVED accesses must preserve the logical `X | R` boundary.
   One-active-axis values may cross that boundary because the remaining axes
   are true broadcasts. Indirect accesses reject.
5. **Normal legality still runs.** A staged proof supplements dependency
   matching through private implementations; ordinary scoring and
   vertical-fusion APIs remain unchanged. Device, mutation, cycle, and backend
   checks still run.
6. **Plans do not cross the mutation boundary.** Fusion-time planning is
   discarded before codegen. Codegen rebuilds from the final fused topology and
   asserts if the accepted staged structure can no longer be reproduced.
7. **Append uses the same path.** `_can_fuse` recognizes an existing nested
   node and derives its prospective append plan locally; `fuse_with`
   independently replans before mutation.
8. **The standalone append is narrow.** Only an exact `FusedStagedReduction`
   may append a parent-stage consumer after full replanning. Subclasses and
   shifted accesses fail closed.

## Intended follow-up

The indexed-forwarding rebase (F1) should record the inherited
parent-to-grouped relations too. At that point the legacy equivalence branch
disappears and every non-exact staged dependency is an exact planner-owned
relation.
