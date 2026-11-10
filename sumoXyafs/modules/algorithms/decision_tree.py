"""Algorithm 6: Decision Tree / Trigger Rule for Migrate vs. Stay decisions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState

try:
    from sklearn.tree import DecisionTreeClassifier  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class DecisionTreeSelector(RSUSelector):
    """Supervised Decision Tree classifier deciding 'Migrate or Stay'.

    Labels past decisions by whether staying caused a deadline miss;
    the fitted tree becomes the trigger rule for proactive migration.
    """

    name: str = "decision_tree"

    def __init__(self, max_depth: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_depth = int(max_depth)
        self.sklearn_tree: DecisionTreeClassifier | None = None
        # Rule thresholds: [q_threshold, dist_threshold, slack_threshold]
        self.rules = {"queue_thresh": 0.8, "dist_thresh": 0.75, "slack_thresh": 0.0}

    def _extract_features(self, state: OffloadingState) -> np.ndarray:
        curr_rsu = state.execution_rsu or state.observed_rsu
        q = state.queue_depths.get(curr_rsu, 0)
        cap = max(1, state.rsu_capacities.get(curr_rsu, 20))
        norm_q = min(1.0, q / cap)
        dist = state.rsu_distances.get(curr_rsu, 1000.0)
        norm_dist = min(1.0, dist / 1500.0)
        norm_speed = min(1.0, state.speed / 30.0)
        norm_slack = max(-1.0, min(1.0, state.deadline_slack_steps / 10.0))
        return np.array([norm_q, norm_dist, norm_speed, norm_slack], dtype=float)

    def should_migrate(self, state: OffloadingState) -> bool:
        feats = self._extract_features(state)
        if self.sklearn_tree is not None and HAS_SKLEARN:
            pred = self.sklearn_tree.predict(feats.reshape(1, -1))[0]
            return bool(pred == 1)

        # Pure-NumPy interpretable decision tree trigger rule:
        # Migrate if queue is saturated, or slack is negative, or far away with tight deadline
        norm_q, norm_dist, norm_speed, norm_slack = feats
        if norm_q >= self.rules["queue_thresh"]:
            return True
        if norm_slack <= self.rules["slack_thresh"]:
            return True
        if norm_dist >= self.rules["dist_thresh"] and norm_slack <= 0.2:
            return True
        return False

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        curr_rsu = state.execution_rsu if state.execution_rsu in candidates else state.observed_rsu

        # Ask tree model: Migrate (1) or Stay (0)?
        if not self.should_migrate(state):
            return curr_rsu

        # Trigger migration: choose the best alternative non-overloaded RSU
        viable = [r for r in candidates if r != curr_rsu]
        if not viable:
            return curr_rsu

        def alt_score(r: str) -> float:
            d = state.rsu_distances.get(r, 1000.0) / 1500.0
            q = state.queue_depths.get(r, 0) / max(1, state.rsu_capacities.get(r, 20))
            return 0.6 * d + 0.4 * q

        return min(viable, key=alt_score)

    def train_offline(self, records: list[dict]) -> None:
        """Train classifier where target = 1 (staying caused deadline miss), 0 (stayed and met)."""
        X_list: list[list[float]] = []
        y_list: list[int] = []

        for row in records:
            if "deadline_met" not in row:
                continue
            action = str(row.get("action", ""))
            # If the vehicle stayed local or sticky
            is_stay = action in {"keep_local", "sticky", "nominal"}
            if not is_stay and "action" in row:
                continue

            q = float(row.get("queue_depth_action", 0)) / 20.0
            dist = float(row.get("norm_distance_to_action", 0.5))
            speed = min(1.0, float(row.get("speed", 0.0)) / 30.0)
            slack = max(-1.0, min(1.0, float(row.get("deadline_slack_steps", 0)) / 10.0))

            # If staying resulted in deadline miss -> label as 1 (should have migrated)
            target = 1 if int(row["deadline_met"]) == 0 else 0
            X_list.append([q, dist, speed, slack])
            y_list.append(target)

        if len(X_list) < 10:
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if HAS_SKLEARN:
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            tree.fit(X, y)
            self.sklearn_tree = tree
        else:
            # Calibrate threshold rules based on data percentiles
            miss_idx = y == 1
            if np.any(miss_idx):
                self.rules["queue_thresh"] = float(np.percentile(X[miss_idx, 0], 25))
                self.rules["slack_thresh"] = float(np.percentile(X[miss_idx, 3], 75))

    def save(self, file_path: str | Path) -> None:
        data = {"rules": self.rules, "max_depth": self.max_depth}
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.rules = data.get("rules", self.rules)
        self.max_depth = int(data.get("max_depth", self.max_depth))
