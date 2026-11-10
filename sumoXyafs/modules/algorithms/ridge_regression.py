"""Algorithm 3: Ridge Regression for latency prediction and RSU selection."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState


class RidgeRegressionSelector(RSUSelector):
    """Supervised Ridge regression predicting response latency per (vehicle, RSU).

    Picks the candidate RSU with the lowest predicted latency.
    """

    name: str = "ridge"

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        # Default prior weights: [norm_dist, norm_q, norm_speed, compute_s, is_observed, bias]
        self.weights = np.array([1.2, 1.5, 0.1, 1.0, -0.3, 0.5], dtype=float)

    def _extract_pair_features(self, state: OffloadingState, rsu_id: str) -> list[float]:
        dist = state.rsu_distances.get(rsu_id, 1000.0)
        norm_dist = min(1.0, dist / 1500.0)
        q = state.queue_depths.get(rsu_id, 0)
        cap = max(1, state.rsu_capacities.get(rsu_id, 20))
        norm_q = min(1.0, q / cap)
        norm_speed = min(1.0, state.speed / 30.0)
        compute = float(state.remaining_compute_s)
        is_obs = 1.0 if rsu_id == state.observed_rsu else 0.0
        return [norm_dist, norm_q, norm_speed, compute, is_obs, 1.0]

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        best_latency = float("inf")
        best_rsu = candidates[0]

        for rsu_id in candidates:
            x = np.array(self._extract_pair_features(state, rsu_id), dtype=float)
            pred_latency = float(np.dot(self.weights, x))

            # Additional penalty if RSU queue has breached capacity
            q = state.queue_depths.get(rsu_id, 0)
            cap = state.rsu_capacities.get(rsu_id, 20)
            if q >= cap:
                pred_latency += 10.0

            if pred_latency < best_latency:
                best_latency = pred_latency
                best_rsu = rsu_id

        return best_rsu

    def train_offline(self, records: list[dict]) -> None:
        """Fit ridge weights w = (X^T X + alpha*I)^(-1) X^T y."""
        X_list: list[list[float]] = []
        y_list: list[float] = []

        for row in records:
            if "latency_s" not in row:
                continue
            dist = float(row.get("norm_distance_to_action", 0.5))
            q = float(row.get("queue_depth_action", 0))
            norm_q = min(1.0, q / 20.0)
            speed = float(row.get("speed", 0.0))
            norm_speed = min(1.0, speed / 30.0)
            compute = float(row.get("remaining_compute_s", 2.0))
            is_obs = 1.0 if row.get("observed_rsu") == row.get("action_rsu") else 0.0
            feat = [dist, norm_q, norm_speed, compute, is_obs, 1.0]
            lat = float(row["latency_s"])

            X_list.append(feat)
            y_list.append(lat)

        if len(X_list) < 5:
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        d = X.shape[1]
        reg = self.alpha * np.eye(d)
        reg[-1, -1] = 0.0  # Do not regularize bias
        self.weights = np.linalg.solve(X.T @ X + reg, X.T @ y)

    def save(self, file_path: str | Path) -> None:
        data = {"alpha": self.alpha, "weights": self.weights.tolist()}
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.alpha = float(data.get("alpha", self.alpha))
        self.weights = np.array(data.get("weights", self.weights), dtype=float)
