from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import torch.fx as fx
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch._inductor.fx_passes.bucketing import (
    bucket_key,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollBucket,
    CollectiveInfo,
    get_group_name,
    is_compute_node,
)
from torch.utils._ordered_set import OrderedSet


@dataclass
class Event:
    """Represents a point in the timeline with a representative operation."""

    node: fx.Node  # Single representative node
    event_type: Literal["compute", "starts", "waits"]
    position: int
    prev: Optional["Event"] = None
    next: Optional["Event"] = None

    @property
    def is_start(self) -> bool:
        return self.event_type == "starts"

    @property
    def is_wait(self) -> bool:
        return self.event_type == "waits"

    @property
    def is_compute(self) -> bool:
        return self.event_type == "compute"

    def unlink(self) -> tuple[Optional["Event"], Optional["Event"]]:
        """Remove this event from the linked list, return (prev, next)."""
        prev_event, next_event = self.prev, self.next
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        self.prev = None
        self.next = None
        return prev_event, next_event

    def insert_between(self, prev_event: Optional["Event"], next_event: Optional["Event"]) -> None:
        """Insert this event between prev_event and next_event in the linked list."""
        if prev_event:
            prev_event.next = self
        self.prev = prev_event

        if next_event:
            next_event.prev = self
        self.next = next_event


@dataclass
class HidingInterval:
    """Represents a hidden collective operation."""

    coll_start: fx.Node
    coll_wait: fx.Node
    hiding_compute: fx.Node


class OverlapPreservingBucketer:
    """
    Buckets collective operations while preserving compute-collective overlap relationships.
    Uses an augmented graph to track dependencies between compute and collective operations.
    """

    def __init__(
        self,
        graph: fx.Graph,
        collective_info: dict[fx.Node, CollectiveInfo],
        node_ancestors: dict[fx.Node, OrderedSet[fx.Node]],
        scheduled: OrderedSet[fx.Node],
        max_bucket_memory_gb: float = 1.0,
        max_coll_distance: int = 1000,
        insert_overlap_deps: bool = False,
    ):
        self.graph = graph
        self.collective_info = collective_info
        self.node_ancestors = node_ancestors
        self.scheduled = scheduled
        self.max_bucket_memory_gb = max_bucket_memory_gb
        self.node_idx = {n: i for i, n in enumerate(scheduled)}
        self.aug_graph = AugmentedGraphHelper(self.graph, self.node_ancestors)
        self.max_coll_distance = max_coll_distance
        self.insert_overlap_deps = insert_overlap_deps
        self.node_to_event: dict[fx.Node, Event] = {}
        self.pg_to_timeline: dict[str, Optional[Event]] = self.build_timelines()
        self.pg_to_hiding_intervals: dict[str, list[HidingInterval]] = (
            self.build_hiding_intervals()
        )

        self._add_timeline_constraints()

    def build_timelines(self) -> dict[str, Optional[Event]]:
        all_pgs = OrderedSet()
        for start in self.collective_info:
            pg = get_group_name(start)
            all_pgs.add(pg)

        pg_timeline: dict[str, Optional[Event]] = {}
        for pg in all_pgs:
            pg_timeline[pg] = self.build_timeline(pg)

        # Populate node_to_event mapping by traversing linked lists
        for head_event in pg_timeline.values():
            event = head_event
            while event is not None:
                self.node_to_event[event.node] = event
                event = event.next

        return pg_timeline

    def build_timeline(self, pg: str) -> Optional[Event]:
        head = None
        prev_event = None
        position = 0

        for node in self.scheduled:
            node_type = None

            # Determine if this node is relevant for this PG
            if node in self.collective_info and get_group_name(node) == pg:
                node_type = "starts"
            elif is_wait_tensor(node) and get_group_name(node.args[0]) == pg:
                node_type = "waits"
            elif is_compute_node(node):
                node_type = "compute"

            if node_type is None:
                continue

            event = Event(node=node, event_type=node_type, position=position)

            # Link to previous event
            if prev_event:
                event.prev = prev_event
                prev_event.next = event
            else:
                head = event

            prev_event = event
            position += 1

        return head

    def build_hiding_intervals(self) -> dict[str, list[HidingInterval]]:
        """
        Identify all hiding intervals, grouped by process group.
        """
        intervals_by_pg: dict[str, list[HidingInterval]] = defaultdict(list)

        for start, info in self.collective_info.items():
            if info.hiding_node and not info.is_exposed:
                pg = get_group_name(start)
                intervals_by_pg[pg].append(
                    HidingInterval(
                        coll_start=start,
                        coll_wait=info.wait_node,
                        hiding_compute=info.hiding_node,
                    )
                )

        # Sort intervals by their position in the timeline (rescheduled order)
        for pg, intervals in intervals_by_pg.items():
            intervals.sort(
                key=lambda interval: self.node_to_event[interval.coll_start].position
            )

        return intervals_by_pg

    def _add_timeline_constraints(self) -> None:
        """
        Add O(n) constraints per process group:
        1. For each hiding interval: frontier_starts -> compute -> frontier_waits
        2. Build prev/next chains for constrained starts/waits

        Note: We do NOT add global compute order constraints here because they would
        prevent bucketing. Instead, compute order is enforced via additional_deps
        during the final topological sort in bucket_collectives().
        """
        # Add per process group constraints
        for pg in self.pg_to_timeline:
            self._add_pg_timeline_constraints(pg)

    def _add_pg_timeline_constraints(self, pg: str) -> None:
        """Add augmented graph dependencies from the doubly-linked event timeline.

        We encode sequential ordering of all events in the timeline (prev -> next).
        These constraints prevent bucketing initially. During bucketing attempts,
        we unlink events to test if bucketing is valid, then relink if it fails.
        The doubly-linked list tracks positions that can be modified during bucketing.
        """
        hiding_intervals = self.pg_to_hiding_intervals[pg]

        # Add constraints for hiding intervals (start -> compute -> wait)
        for interval in hiding_intervals:
            start_node = interval.coll_start
            compute_node = interval.hiding_compute
            wait_node = interval.coll_wait

            # Enforce: start -> compute -> wait
            self.aug_graph.add_extra_dep(n=compute_node, dep=start_node)
            self.aug_graph.add_extra_dep(n=wait_node, dep=compute_node)

        # Add sequential constraints for the timeline (prev -> next)
        # This encodes the initial ordering and prevents bucketing until we unlink
        head_event = self.pg_to_timeline[pg]
        event = head_event
        while event is not None and event.next is not None:
            # Add dependency: next depends on current
            self.aug_graph.add_extra_dep(n=event.next.node, dep=event.node)
            event = event.next

    def bucket_collectives(self) -> None:
        """Main entry point for bucketing collectives."""

        # Group collectives by bucket key (type, group, etc.)
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for start in self.collective_info:
            key = bucket_key(start)
            if key is not None:
                grouped_collectives[key].add(start)

        all_buckets: list[CollBucket] = []
        for collective_group in grouped_collectives.values():
            buckets = self._find_buckets(collective_group)
            all_buckets.extend(buckets)

        # Apply bucketing transformations
        # Dependencies are tracked in aug_graph.extra_deps during bucketing
        for coll_bucket in all_buckets:
            if len(coll_bucket.collectives) <= 1:
                continue

            self._apply_bucket(coll_bucket)

        # Extract all dependencies from augmented graph
        additional_deps = self.aug_graph.get_all_extra_deps()

        # Add compute ordering
        comp_nodes = [n for n in self.scheduled if is_compute_node(n)]
        for i in range(len(comp_nodes) - 1):
            additional_deps[comp_nodes[i + 1]].add(comp_nodes[i])

        # Apply topological sort with all dependencies
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        _stable_topological_sort(self.graph, additional_deps)

        # After topological sort, preserve dependencies using effect tokens
        if self.insert_overlap_deps:
            self._preserve_dependencies_with_tokens(additional_deps)

        self.graph.lint()

    def _find_buckets(
        self,
        collective_group: OrderedSet[fx.Node],
    ) -> list[CollBucket]:
        """Find valid buckets within a group of similar collectives."""

        max_bucket_bytes = int(self.max_bucket_memory_gb * 1024 * 1024 * 1024)
        buckets = []
        processed: OrderedSet[fx.Node] = OrderedSet()

        for start_node in collective_group:
            if start_node in processed:
                continue

            # Initialize bucket with first collective
            bucket_info = CollBucket(
                collectives=[start_node],
                total_bytes=self.collective_info[start_node].size_bytes,
            )
            processed.add(start_node)
            start_node_idx = self.node_idx[start_node]

            # TODO - limit within range
            for candidate in collective_group:
                if candidate in processed:
                    continue

                candidate_idx = self.node_idx[candidate]
                # Check if candidate is within max distance from the bucket start
                if abs(candidate_idx - start_node_idx) > self.max_coll_distance:
                    continue

                candidate_bytes = self.collective_info[candidate].size_bytes
                if bucket_info.total_bytes + candidate_bytes > max_bucket_bytes:
                    continue

                if self._can_add_to_bucket(bucket_info, candidate):
                    bucket_info.collectives.append(candidate)
                    bucket_info.total_bytes += candidate_bytes
                    processed.add(candidate)

            if len(bucket_info.collectives) > 1:
                buckets.append(bucket_info)

        return buckets

    def _ancestor_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if there's an ancestor relationship between two nodes."""
        return n1 in self.node_ancestors[n2] or n2 in self.node_ancestors[n1]

    def _should_skip_event(self, event: Event, bucketed_colls: list[fx.Node]) -> bool:
        """Check if event should be skipped (belongs to bucketed collectives)."""
        if event.node in bucketed_colls:
            return True
        if event.is_wait:
            # Check if this wait's start is one of the bucketed collectives
            wait_node = event.node
            if hasattr(wait_node, 'args') and wait_node.args[0] in bucketed_colls:
                return True
        return False

    def _get_compute_interval(self, event: Event) -> Optional[tuple[int, int]]:
        """Get (start, compute) interval if event is a start with hiding compute."""
        if not event.is_start:
            return None
        coll = event.node
        if coll in self.collective_info:
            if hiding_node := self.collective_info[coll].hiding_node:
                return (event.position, self.node_to_event[hiding_node].position)
        return None

    def _get_execution_interval(self, event: Event) -> Optional[tuple[int, int]]:
        """Get (start, wait) execution interval if event is a start."""
        if not event.is_start:
            return None
        coll = event.node
        if coll in self.collective_info:
            wait = self.collective_info[coll].wait_node
            return (event.position, self.node_to_event[wait].position)
        return None

    def _check_interval_violations(
        self,
        event: Event,
        bucket_execution_interval: tuple[int, int],
        bucket_hiding_compute_positions: list[int],
    ) -> bool:
        """
        Check if event creates interval violations with bucket.
        Returns True if there's a violation.
        """
        def enclosed_interval(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
            return outer[0] < inner[0] and inner[1] < outer[1]

        bucket_execution_start, bucket_execution_end = bucket_execution_interval

        # Check compute intervals
        if compute_interval := self._get_compute_interval(event):
            # Would our execution interval enclose their compute interval?
            if enclosed_interval(compute_interval, (bucket_execution_start, bucket_execution_end)):
                return True

        # Check execution intervals
        if event.is_start:
            if execution_interval := self._get_execution_interval(event):
                # Would their execution interval enclose any of our compute intervals?
                for compute_pos in bucket_hiding_compute_positions:
                    compute_interval = (bucket_execution_start, compute_pos)
                    if enclosed_interval(compute_interval, execution_interval):
                        return True

        return False

    def _walk_timeline_checking_violations(
        self,
        start_event: Optional[Event],
        end_position: int,
        direction: Literal["forward", "backward"],
        bucket_execution_interval: tuple[int, int],
        bucket_hiding_compute_positions: list[int],
        all_bucketed_colls: list[fx.Node],
    ) -> bool:
        """
        Walk timeline in given direction checking for interval violations.
        Returns True if a violation is found, False otherwise.

        Args:
            start_event: First event to check (or None if no walk needed)
            end_position: Position to stop at (exclusive)
            direction: 'forward' or 'backward'
            bucket_execution_interval: Execution interval of the bucket
            bucket_hiding_compute_positions: Compute positions to check
            all_bucketed_colls: Collectives being bucketed (to skip)
        """
        curr_event = start_event

        while curr_event is not None:
            # Check if we've reached the end position
            if direction == "forward":
                if curr_event.position >= end_position:
                    break
            else:  # backward
                if curr_event.position <= end_position:
                    break

            # Skip events belonging to bucketed collectives
            if not self._should_skip_event(curr_event, all_bucketed_colls):
                # Check for interval violations
                if self._check_interval_violations(
                    curr_event, bucket_execution_interval, bucket_hiding_compute_positions
                ):
                    return True

            # Move to next event in the direction
            curr_event = curr_event.next if direction == "forward" else curr_event.prev

        return False

    def _preserves_hiding_intervals(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
    ) -> bool:
        """
        Check that (start_pos, wait_pos) doesn't violate any hiding intervals or collectives.

        Walks the timeline between new and original positions, checking:
        1. All bucket hiding compute stays between new start/wait
        2. No other collective's compute interval is enclosed by bucket execution interval
        3. No other collective's execution interval encloses bucket compute intervals
        """
        # Collect all collectives being bucketed
        all_bucketed_colls = [candidate] + list(bucket_info.collectives)

        # Collect hiding compute positions for the bucket
        bucket_hiding_compute_positions = []
        for coll in all_bucketed_colls:
            if hiding_node := self.collective_info[coll].hiding_node:
                bucket_hiding_compute_positions.append(
                    self.node_to_event[hiding_node].position
                )

        # Get new positions
        new_start_event = self.node_to_event[start_pos]
        new_wait_event = self.node_to_event[wait_pos]

        # Check 1: All bucket hiding compute must be between new start and wait
        for compute_pos in bucket_hiding_compute_positions:
            if not (new_start_event.position < compute_pos < new_wait_event.position):
                return False

        # Get original positions
        candidate_start_event = self.node_to_event[candidate]
        bucket_start_event = self.node_to_event[bucket_info.collectives[0]]
        candidate_wait_event = self.node_to_event[self.collective_info[candidate].wait_node]
        bucket_wait_event = self.node_to_event[self.collective_info[bucket_info.collectives[0]].wait_node]

        # Latest start and earliest wait among collectives being bucketed
        latest_start_pos = max(candidate_start_event.position, bucket_start_event.position)
        earliest_wait_pos = min(candidate_wait_event.position, bucket_wait_event.position)

        # Bucket execution interval
        bucket_execution_interval = (new_start_event.position, new_wait_event.position)

        # Check 2: Walk timeline forward from new_start to latest_start
        if new_start_event.position < latest_start_pos:
            if self._walk_timeline_checking_violations(
                new_start_event.next,
                latest_start_pos,
                "forward",
                bucket_execution_interval,
                bucket_hiding_compute_positions,
                all_bucketed_colls,
            ):
                return False

        # Check 3: Walk timeline backward from new_wait to earliest_wait
        if new_wait_event.position > earliest_wait_pos:
            if self._walk_timeline_checking_violations(
                new_wait_event.prev,
                earliest_wait_pos,
                "backward",
                bucket_execution_interval,
                bucket_hiding_compute_positions,
                all_bucketed_colls,
            ):
                return False

        return True

    def remove_from_event(self, node: fx.Node) -> tuple[Optional[Event], Optional[Event]]:
        """Remove node from timeline and return (prev_event, next_event)."""
        event = self.node_to_event[node]
        assert not event.is_compute, "Cannot remove compute events from timeline"

        prev_event, next_event = event.unlink()

        # Remove augmented graph dependency
        if prev_event:
            self.aug_graph.remove_extra_dep(n=node, dep=prev_event.node)
        if next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=node)

        # Add bypass dependency
        if prev_event and next_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=prev_event.node)

        return prev_event, next_event

    def restore_to_event(
        self, node: fx.Node, prev_event: Optional[Event], next_event: Optional[Event]
    ) -> None:
        """Restore node to timeline after failed merge attempt."""
        event = self.node_to_event[node]

        # Reinsert into linked list
        event.insert_between(prev_event, next_event)
        if prev_event:
            self.aug_graph.add_extra_dep(n=node, dep=prev_event.node)
        if next_event and not prev_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=node)

        # Remove bypass dependency
        if prev_event and next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=prev_event.node)

    def _try_rail_position(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
    ) -> bool:
        """
        Try a specific rail position for the candidate.
        Returns True if valid and merges are successful.
        """
        candidate_info = self.collective_info[candidate]
        candidate_start = candidate
        candidate_wait = candidate_info.wait_node

        # Quick check: does this violate hiding intervals?
        if not self._preserves_hiding_intervals(
            bucket_info, candidate, start_pos, wait_pos
        ):
            return False

        # Determine which start needs to move
        existing_coll = bucket_info.collectives[0]
        if start_pos == existing_coll:
            start_to_move = candidate
        else:
            assert start_pos == candidate
            start_to_move = existing_coll

        # Remove start from timeline
        start_prev, start_next = self.remove_from_event(start_to_move)

        # Check if starts can be merged
        if self.aug_graph.has_path(existing_coll, candidate) or self.aug_graph.has_path(
            candidate, existing_coll
        ):
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            return False

        # Merge starts
        self.aug_graph.merge_to_set(existing_coll, candidate)

        # Determine which wait needs to move
        existing_wait = self.collective_info[existing_coll].wait_node
        candidate_wait = self.collective_info[candidate].wait_node

        if wait_pos == existing_wait:
            wait_to_move = candidate_wait
        else:
            wait_to_move = existing_wait

        # Remove wait from timeline
        wait_prev, wait_next = self.remove_from_event(wait_to_move)

        # Check if waits can be merged
        if self.aug_graph.has_path(
            existing_wait, candidate_wait
        ) or self.aug_graph.has_path(candidate_wait, existing_wait):
            # Restore wait constraints
            self.restore_to_event(wait_to_move, wait_prev, wait_next)
            # Unmerge the start we just merged
            self.aug_graph.unmerge_node(candidate)
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            return False

        # Merge waits - success!
        self.aug_graph.merge_to_set(existing_wait, candidate_wait)

        # Update node_to_event for moved nodes
        target_start_event = self.node_to_event[start_pos]
        target_wait_event = self.node_to_event[wait_pos]

        self.node_to_event[candidate] = target_start_event
        self.node_to_event[candidate_wait] = target_wait_event

        return True

    def _has_ancestor_conflicts(self, bucket_info: CollBucket, candidate: fx.Node) -> bool:
        """
        Check if candidate has ancestor conflicts with bucket collectives.
        Returns True if there are conflicts.
        """
        candidate_info = self.collective_info[candidate]
        candidate_wait = candidate_info.wait_node

        for coll in bucket_info.collectives:
            # Check if collectives are ancestors of each other
            if self._ancestor_dep(coll, candidate):
                return True

            # Check if waits are ancestors of each other
            coll_wait = self.collective_info[coll].wait_node
            if self._ancestor_dep(candidate_wait, coll_wait):
                return True

            # Check if existing hiding node conflicts with candidate wait
            if hiding_node := self.collective_info[coll].hiding_node:
                if self._ancestor_dep(hiding_node, candidate_wait):
                    return True

            # Check if candidate hiding node conflicts with existing wait
            if new_hiding_node := candidate_info.hiding_node:
                if self._ancestor_dep(new_hiding_node, coll_wait):
                    return True

        return False

    def _can_add_to_bucket(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
    ) -> bool:
        """
        Check if candidate can be added to bucket without interfering
        with comm/compute overlap.
        """
        candidate_info = self.collective_info[candidate]

        # Step 1: Quick check using precomputed ancestors
        # These ancestors are computed prior to adding augmented dependencies and not updated,
        # so if any of these checks fail then the merge will not be topologically valid
        # even ignoring comm/compute overlap
        if self._has_ancestor_conflicts(bucket_info, candidate):
            return False

        # Step 2: Try different rail positions
        existing_coll = bucket_info.collectives[0]
        existing_wait = self.collective_info[existing_coll].wait_node

        candidate_start = candidate
        candidate_wait = candidate_info.wait_node

        # Try combinations in order of likelihood to succeed
        # (early start, later wait is most likely to work)
        combinations = [
            (
                existing_coll,
                candidate_wait,
            ),  # Move candidate start early, keep wait late
            (
                existing_coll,
                existing_wait,
            ),  # Move candidate start early, move wait early
            (candidate_start, candidate_wait),  # Keep both in place
            (candidate_start, existing_wait),  # Keep start in place, move wait early
        ]

        for start_pos, wait_pos in combinations:
            if self._try_rail_position(bucket_info, candidate, start_pos, wait_pos):
                return True

        return False

    def _apply_bucket(
        self, bucket_info: CollBucket
    ) -> tuple[fx.Node, fx.Node]:
        """
        Apply bucketing transformation.

        Dependencies are added to aug_graph.extra_deps and transferred from old nodes.
        Returns (new_start, new_wait).
        """

        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_reduce_scatter_bucket,
        )

        bucket = bucket_info.collectives

        # Collect old nodes BEFORE they're erased
        old_starts = list(bucket)
        old_waits = [self.collective_info[n].wait_node for n in bucket]

        # Find where to place the bucketed operations
        next_node = bucket[0]
        while next_node in bucket:
            next_node = next_node.next

        # Don't use wait_insertion_point - let merge functions place waits naturally
        # The wait_insertion_point feature tries to move waits to a specific location,
        # but this can cause issues when that location is one of the nodes being erased
        # Create bucketed collective (this will erase old nodes)
        if is_all_gather(bucket[0]):
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode="custom_ops",
            )
        else:
            assert is_reduce_scatter(bucket[0])
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode="custom_ops",
            )

        # Get new nodes
        new_waits = [n for n in new_nodes if is_wait_tensor(n)]
        assert len(new_waits) == 1

        new_wait = new_waits[0]
        new_start = new_wait.args[0]
        assert isinstance(new_start, fx.Node)

        # Transfer all dependencies from old nodes to new nodes
        # This must happen after merge but old nodes are already erased, so we rely on
        # extra_deps dict which still has their entries
        for old_start in old_starts:
            self.aug_graph.transfer_erased_node_deps(old_start, new_start)
        for old_wait in old_waits:
            self.aug_graph.transfer_erased_node_deps(old_wait, new_wait)

        # Add hiding interval constraints to augmented graph
        for coll in bucket:
            info = self.collective_info[coll]
            if info.hiding_node and not info.is_exposed:
                # Compute depends on collective start
                self.aug_graph.add_extra_dep(n=info.hiding_node, dep=new_start)
                # Wait depends on compute
                self.aug_graph.add_extra_dep(n=new_wait, dep=info.hiding_node)

        return new_start, new_wait

    def _preserve_dependencies_with_tokens(
        self, additional_deps: dict[fx.Node, OrderedSet[fx.Node]]
    ) -> None:
        """
        Preserve dependencies using effect tokens and with_effects higher-order op.

        Uses the standalone token_dependencies utility for consistent behavior
        across different overlap scheduling approaches.
        """
        from torch._inductor.fx_passes.control_dependencies import (
            preserve_node_ordering,
        )

        preserve_node_ordering(self.graph, additional_deps)
