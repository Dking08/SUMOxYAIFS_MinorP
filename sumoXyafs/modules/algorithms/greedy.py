"""Greedy baseline RSU offloading and sticky-handover algorithm."""
from __future__ import annotations

import math
from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState


class GreedyBaselineSelector(RSUSelector):
    """Baseline heuristic: deadline-aware sticky handover + nearest non-overloaded offload."""

    name: str = "greedy"

    def __init__(
        self,
        overload_threshold: int = 20,
        distance_latency_norm_m: float = 500.0,
        distance_latency_max_s: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.overload_threshold = max(1, int(overload_threshold))
        self.distance_latency_norm_m = max(1.0, float(distance_latency_norm_m))
        self.distance_latency_max_s = max(0.0, float(distance_latency_max_s))

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        target_rsu = state.observed_rsu

        # 1. Sticky handover check when observed RSU has changed
        if state.observed_rsu != state.owner_rsu and state.execution_rsu in candidates:
            curr_exec = state.execution_rsu
            queue_depth = state.queue_depths.get(curr_exec, 0)
            dist = state.rsu_distances.get(curr_exec, 0.0)
            net_lat = self._dist_latency(dist)
            est_finish = state.step + int(math.ceil(queue_depth + state.remaining_compute_s + net_lat))
            if est_finish <= state.deadline_step:
                target_rsu = curr_exec
            else:
                target_rsu = state.observed_rsu

        # 2. Overload check
        if self._is_overloaded(target_rsu, state):
            alt = self._nearest_available(state, exclude={target_rsu})
            if alt is not None:
                target_rsu = alt

        return target_rsu

    def _dist_latency(self, distance_m: float) -> float:
        if self.distance_latency_max_s <= 0.0:
            return 0.0
        return self.distance_latency_max_s * (distance_m / self.distance_latency_norm_m)

    def _is_overloaded(self, rsu_id: str, state: OffloadingState) -> bool:
        q = state.queue_depths.get(rsu_id, 0)
        cap = state.rsu_capacities.get(rsu_id, 20)
        limit = min(self.overload_threshold, cap)
        return q >= limit

    def _nearest_available(self, state: OffloadingState, exclude: set[str]) -> str | None:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        best_rsu = None
        best_dist = float("inf")
        for rsu_id in candidates:
            if rsu_id in exclude or self._is_overloaded(rsu_id, state):
                continue
            dist = state.rsu_distances.get(rsu_id, float("inf"))
            if dist < best_dist:
                best_dist = dist
                best_rsu = rsu_id
        return best_rsu
