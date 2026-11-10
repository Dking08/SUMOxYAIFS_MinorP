"""Algorithm 5: Markov Chain for proactive RSU migration based on mobility transitions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import DecisionFeedback, OffloadingState


class MarkovChainSelector(RSUSelector):
    """Statistical Markov Chain predicting the vehicle's next RSU from mobility traces.

    Proactively migrates the service to the predicted next RSU when the vehicle
    is approaching the boundary of its current RSU coverage.
    """

    name: str = "markov"

    def __init__(self, proactive_distance_m: float = 600.0, min_transition_prob: float = 0.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proactive_distance_m = float(proactive_distance_m)
        self.min_transition_prob = float(min_transition_prob)
        # Transition counts: counts[from_rsu][to_rsu]
        self.counts: dict[str, dict[str, float]] = {}
        # Track last observed RSU per vehicle to record transitions online
        self._last_observed: dict[str, str] = {}

    def _record_transition(self, src: str, dst: str) -> None:
        if not src or not dst or src == dst:
            return
        row = self.counts.setdefault(src, {})
        row[dst] = row.get(dst, 0.0) + 1.0

    def predict_next_rsu(self, curr_rsu: str, candidates: list[str]) -> tuple[str | None, float]:
        row = self.counts.get(curr_rsu, {})
        if not row:
            return None, 0.0

        total = sum(row.values()) + len(candidates) * 0.1
        best_rsu = None
        best_prob = 0.0

        for rsu_id in candidates:
            if rsu_id == curr_rsu:
                continue
            prob = (row.get(rsu_id, 0.0) + 0.1) / total
            if prob > best_prob:
                best_prob = prob
                best_rsu = rsu_id

        return best_rsu, best_prob

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        curr_rsu = state.observed_rsu
        # Online tracking of vehicle transitions
        prev = self._last_observed.get(state.vehicle_id)
        if prev and prev != curr_rsu:
            self._record_transition(prev, curr_rsu)
        self._last_observed[state.vehicle_id] = curr_rsu

        dist_to_curr = state.rsu_distances.get(curr_rsu, 0.0)
        is_moving_away = dist_to_curr >= self.proactive_distance_m and state.speed > 5.0

        # If vehicle is moving away from current RSU, check proactive migration
        if is_moving_away:
            predicted_rsu, prob = self.predict_next_rsu(curr_rsu, candidates)
            if predicted_rsu is not None and prob >= self.min_transition_prob:
                q = state.queue_depths.get(predicted_rsu, 0)
                cap = state.rsu_capacities.get(predicted_rsu, 20)
                # Check that predicted RSU is not overloaded
                if q < cap:
                    return predicted_rsu

        # Otherwise remain at observed RSU (or sticky current execution RSU if not overloaded)
        exec_rsu = state.execution_rsu
        if exec_rsu in candidates:
            q_exec = state.queue_depths.get(exec_rsu, 0)
            cap_exec = state.rsu_capacities.get(exec_rsu, 20)
            if q_exec < cap_exec and state.rsu_distances.get(exec_rsu, 1000.0) < self.proactive_distance_m:
                return exec_rsu

        return curr_rsu

    def update(self, feedback: DecisionFeedback) -> None:
        pass

    def train_offline(self, records: list[dict]) -> None:
        """Count transitions from sequential vehicle telemetry or decision records."""
        by_vehicle: dict[str, list[tuple[int, str]]] = {}
        for row in records:
            vid = str(row.get("vehicle_id", ""))
            step = int(row.get("step", 0))
            rsu = str(row.get("observed_rsu", ""))
            if vid and rsu:
                by_vehicle.setdefault(vid, []).append((step, rsu))

        for vid, steps in by_vehicle.items():
            steps.sort(key=lambda x: x[0])
            for i in range(len(steps) - 1):
                s1, r1 = steps[i]
                s2, r2 = steps[i + 1]
                if r1 != r2:
                    self._record_transition(r1, r2)

    def save(self, file_path: str | Path) -> None:
        data = {
            "proactive_distance_m": self.proactive_distance_m,
            "min_transition_prob": self.min_transition_prob,
            "counts": self.counts,
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.proactive_distance_m = float(data.get("proactive_distance_m", self.proactive_distance_m))
        self.min_transition_prob = float(data.get("min_transition_prob", self.min_transition_prob))
        self.counts = data.get("counts", {})
