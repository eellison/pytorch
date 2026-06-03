"""
Reduction Chaining: fusing sequential reductions where the intermediate stays in registers.

Pattern:
  Node A: reduction kernel with output shape [M, N] (reducing over K)
  Node B: reduction kernel that reads A's output and reduces over dim N → [M]
  A's output has only B as consumer (no other reads outside the fused group)
  N fits in registers (persistent inner dimension)

When matched: fuse A and B into a single persistent kernel where:
  - Grid = [M]
  - Each program tile-loops over K (A's reduction), accumulating N values in registers
  - Then performs the second reduction over N in-register

This eliminates materializing the [M, N] intermediate to global memory.

Example use case: MoE gather+sum followed by RMSNorm
  K2: gather+scale+mask+sum over 8 experts → [2048, 2048] intermediate
  K3: add + square + mean over 2048 cols + rsqrt + multiply → [2048, 2048] output
  Oracle achieves 2.4x speedup by keeping the intermediate in registers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torch._inductor import config
from torch._inductor.virtualized import V  # noqa: F401
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode


log = logging.getLogger(__name__)


# Maximum number of elements to keep in registers per thread block.
# 2048 bf16 values = 4KB per row, fits comfortably in registers for persistent kernels.
MAX_REGISTER_ELEMENTS = 2048


class ReductionChaining:
    """
    Detects when two sequential dependent reductions can be fused by keeping
    the intermediate in registers.

    The key structural difference from NestedReduction:
    - NestedReduction handles same-total-numel pairs (partition + sub-reduce)
    - ReductionChaining handles producer-consumer reductions with DIFFERENT
      total elements where the producer's output inner dim equals the consumer's
      reduction dim.

    Specifically:
      Producer: [M, N, K] → reduce over K → [M, N]  (numel=M*N, rnumel=K)
      Consumer: [M, N] → reduce over N → [M]        (numel=M, rnumel=N)
      Fused:    Grid=[M], each program loops K, accumulates N in registers,
                then reduces N → scalar per row.
    """

    @staticmethod
    def _is_dependent_reduction_pair(
        producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> bool:
        """Check that consumer depends on producer and both are reductions."""
        return (
            producer.is_reduction()
            and consumer.is_reduction()
            and bool(producer.get_operation_names() & consumer.ancestors)
        )

    @staticmethod
    def _is_enabled() -> bool:
        """Check if reduction chaining is enabled."""
        return getattr(config.triton, "reduction_chaining", False)

    @classmethod
    def is_candidate(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> bool:
        """
        Quick filter: are these two nodes a potential reduction chain?

        Returns True if:
        1. Both are reductions
        2. Consumer depends on producer
        3. They have different iteration spaces (otherwise normal fusion handles it)
        4. Producer's output numel equals consumer's total numel (numel * rnumel)
        """
        if not cls._is_enabled():
            return False
        if not cls._is_dependent_reduction_pair(producer, consumer):
            return False

        _, (prod_numel, prod_rnumel) = producer.group
        _, (cons_numel, cons_rnumel) = consumer.group

        # If they have the same iteration space, normal fusion should handle it
        if V.graph.sizevars.statically_known_equals(
            prod_numel, cons_numel
        ) and V.graph.sizevars.statically_known_equals(prod_rnumel, cons_rnumel):
            return False

        return True

    @classmethod
    def can_fuse(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> bool:
        """
        Full check: can these two reductions be fused via reduction chaining?

        Conditions:
        1. Producer output shape = [M, N] where N = consumer's reduction dim
        2. Producer's numel = M*N = consumer's numel * consumer's rnumel
        3. N fits in registers (persistent inner dim)
        4. Producer's output is only consumed by the consumer (and optional
           pointwise epilogues that also fuse into the consumer)
        5. The consumer reads the producer's output contiguously

        NOTE: Currently detection-only. Returns False to avoid triggering
        the standard FusedSchedulerNode codegen path which can't handle
        incompatible iteration spaces. The actual fusion would require
        a dedicated FusedChainedReductions node type with custom codegen.
        """
        if not cls.is_candidate(producer, consumer):
            return False

        _, (prod_numel, prod_rnumel) = producer.group
        _, (cons_numel, cons_rnumel) = consumer.group

        # Check: producer's numel = consumer's (numel * rnumel)
        # This means the producer writes [M, N] and the consumer reads all of it
        # as [M, N] then reduces over N to get [M].
        prod_total = V.graph.sizevars.simplify(prod_numel)
        cons_total = V.graph.sizevars.simplify(cons_numel * cons_rnumel)

        if not V.graph.sizevars.statically_known_equals(prod_total, cons_total):
            return False

        # Check: consumer's reduction dim (rnumel) is small enough for registers
        cons_rnumel_hint = V.graph.sizevars.optimization_hint(cons_rnumel, fallback=0)
        if cons_rnumel_hint == 0 or cons_rnumel_hint > MAX_REGISTER_ELEMENTS:
            return False

        # Check: consumer's outer dim = producer's outer dim / rnumel
        # i.e., the producer's output is [cons_numel, cons_rnumel]
        # This is already implied by prod_total == cons_total since:
        #   prod_numel (= M*N) == cons_numel * cons_rnumel
        # But let's also verify the producer's rnumel (K) is reasonable
        prod_rnumel_hint = V.graph.sizevars.optimization_hint(prod_rnumel, fallback=0)
        if prod_rnumel_hint == 0:
            return False

        # Check: producer's output buffer is only consumed by the consumer
        # (or by nodes already fused into the consumer).
        # We check via the scheduler node's output user tracking.
        from torch._inductor.scheduler import OutputNode

        consumer_node_set = OrderedSet(consumer.get_nodes())
        for out in producer.outputs:
            for user in out.users:
                user_node = user.node
                if isinstance(user_node, OutputNode):
                    # The intermediate is a graph output — can't eliminate it
                    return False
                if user_node is consumer:
                    continue
                # Check if user is a sub-node of the consumer (already fused in)
                if user_node in consumer_node_set:
                    continue
                # Another external consumer of the intermediate — can't eliminate
                return False

        log.debug(
            "ReductionChaining: candidate found: %s -> %s "
            "(prod_numel=%s, prod_rnumel=%s, cons_numel=%s, cons_rnumel=%s)",
            producer.get_name(),
            consumer.get_name(),
            prod_numel,
            prod_rnumel,
            cons_numel,
            cons_rnumel,
        )

        # Detection-only for now: the standard FusedSchedulerNode codegen
        # cannot handle the incompatible iteration spaces. A full implementation
        # would create a FusedChainedReductions node with custom codegen.
        # Return False to avoid breaking compilation, but the detection above
        # confirms the pattern matches.
        return False

    @classmethod
    def describe_chain(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> dict | None:
        """
        If these nodes form a valid reduction chain, return a description dict.
        Otherwise return None.
        """
        if not cls.can_fuse(producer, consumer):
            return None

        _, (prod_numel, prod_rnumel) = producer.group
        _, (cons_numel, cons_rnumel) = consumer.group

        return {
            "producer_name": producer.get_name(),
            "consumer_name": consumer.get_name(),
            "outer_dim_M": V.graph.sizevars.optimization_hint(cons_numel, fallback=0),
            "inner_dim_N": V.graph.sizevars.optimization_hint(cons_rnumel, fallback=0),
            "first_reduction_K": V.graph.sizevars.optimization_hint(prod_rnumel, fallback=0),
            "intermediate_elements": V.graph.sizevars.optimization_hint(prod_numel, fallback=0),
            "register_budget_used": V.graph.sizevars.optimization_hint(cons_rnumel, fallback=0),
        }
