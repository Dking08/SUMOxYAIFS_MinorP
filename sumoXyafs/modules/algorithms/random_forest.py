"""Algorithm 4: Random Forest Classifier for deadline-miss filtering."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class SimpleTree:
    """Lightweight pure-NumPy decision tree stump/branch for CPU fallback."""

    def __init__(self, feature_idx: int = 0, threshold: float = 0.5, left_val: float = 0.0, right_val: float = 1.0) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val

    def predict(self, x: np.ndarray) -> float:
        val = x[self.feature_idx]
        return self.left_val if val <= self.threshold else self.right_val


class RandomForestSelector(RSUSelector):
    """Supervised Random Forest Classifier predicting deadline-miss probability.

    Filters out RSUs with high predicted miss probability, then picks greedily
    among the remaining candidates.
    """

    name: str = "random_forest"

    def __init__(self, miss_threshold: float = 0.5, n_estimators: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.miss_threshold = float(miss_threshold)
        self.n_estimators = int(n_estimators)
        self.sklearn_model: RandomForestClassifier | None = None
        self.numpy_trees: list[SimpleTree] = []
        self._init_prior_trees()

    def _init_prior_trees(self) -> None:
        # Default physical heuristics encoded as randomized tree ensemble:
        # Features: [0: norm_dist, 1: norm_q, 2: norm_speed, 3: norm_slack, 4: is_observed]
        self.numpy_trees = [
            SimpleTree(feature_idx=1, threshold=0.85, left_val=0.1, right_val=0.9),  # high queue -> miss
            SimpleTree(feature_idx=3, threshold=-0.1, left_val=0.85, right_val=0.15),  # tight slack -> miss
            SimpleTree(feature_idx=0, threshold=0.75, left_val=0.2, right_val=0.7),  # far distance -> miss
            SimpleTree(feature_idx=1, threshold=0.6, left_val=0.15, right_val=0.65),
            SimpleTree(feature_idx=3, threshold=0.2, left_val=0.6, right_val=0.1),
        ]

    def _extract_features(self, state: OffloadingState, rsu_id: str) -> np.ndarray:
        dist = state.rsu_distances.get(rsu_id, 1000.0)
        norm_dist = min(1.0, dist / 1500.0)
        q = state.queue_depths.get(rsu_id, 0)
        cap = max(1, state.rsu_capacities.get(rsu_id, 20))
        norm_q = min(1.0, q / cap)
        norm_speed = min(1.0, state.speed / 30.0)
        norm_slack = max(-1.0, min(1.0, state.deadline_slack_steps / 10.0))
        is_obs = 1.0 if rsu_id == state.observed_rsu else 0.0
        return np.array([norm_dist, norm_q, norm_speed, norm_slack, is_obs], dtype=float)

    def predict_miss_prob(self, features: np.ndarray) -> float:
        if self.sklearn_model is not None and HAS_SKLEARN:
            proba = self.sklearn_model.predict_proba(features.reshape(1, -1))[0]
            # Probability of class 1 (deadline miss)
            return float(proba[1]) if len(proba) > 1 else float(proba[0])

        # NumPy ensemble prediction
        preds = [tree.predict(features) for tree in self.numpy_trees]
        return float(np.mean(preds))

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        rsu_miss_probs: dict[str, float] = {}
        for rsu_id in candidates:
            feats = self._extract_features(state, rsu_id)
            rsu_miss_probs[rsu_id] = self.predict_miss_prob(feats)

        # Filter out RSUs with high predicted miss probability
        viable = [r for r in candidates if rsu_miss_probs[r] <= self.miss_threshold]
        if not viable:
            # If all candidates breach the threshold, pick the one with lowest miss probability
            return min(candidates, key=lambda r: rsu_miss_probs[r])

        # Pick greedily among the viable candidates (prefer least loaded, closest)
        def score(r: str) -> float:
            d = state.rsu_distances.get(r, 1000.0) / 1500.0
            q = state.queue_depths.get(r, 0) / max(1, state.rsu_capacities.get(r, 20))
            return 0.5 * d + 0.5 * q

        return min(viable, key=score)

    def train_offline(self, records: list[dict]) -> None:
        X_list: list[list[float]] = []
        y_list: list[int] = []

        for row in records:
            if "deadline_met" not in row:
                continue
            dist = float(row.get("norm_distance_to_action", 0.5))
            q = float(row.get("queue_depth_action", 0))
            norm_q = min(1.0, q / 20.0)
            speed = min(1.0, float(row.get("speed", 0.0)) / 30.0)
            slack = max(-1.0, min(1.0, float(row.get("deadline_slack_steps", 0)) / 10.0))
            is_obs = 1.0 if row.get("observed_rsu") == row.get("action_rsu") else 0.0
            # Target: 1 if deadline missed, 0 if met
            miss = 1 - int(row["deadline_met"])

            X_list.append([dist, norm_q, speed, slack, is_obs])
            y_list.append(miss)

        if len(X_list) < 10:
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if HAS_SKLEARN:
            model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=5, random_state=42)
            model.fit(X, y)
            self.sklearn_model = model
        else:
            # Train pure-NumPy randomized tree ensemble
            np.random.seed(42)
            trees: list[SimpleTree] = []
            n_samples = len(X)
            for _ in range(self.n_estimators):
                idx = np.random.choice(n_samples, size=n_samples, replace=True)
                feat = np.random.choice(X.shape[1])
                thresh = float(np.median(X[idx, feat]))
                left_mask = X[idx, feat] <= thresh
                right_mask = ~left_mask
                left_val = float(np.mean(y[idx][left_mask])) if np.any(left_mask) else 0.5
                right_val = float(np.mean(y[idx][right_mask])) if np.any(right_mask) else 0.5
                trees.append(SimpleTree(feature_idx=feat, threshold=thresh, left_val=left_val, right_val=right_val))
            self.numpy_trees = trees

    def save(self, file_path: str | Path) -> None:
        data = {
            "miss_threshold": self.miss_threshold,
            "trees": [
                {
                    "feature_idx": t.feature_idx,
                    "threshold": t.threshold,
                    "left_val": t.left_val,
                    "right_val": t.right_val,
                }
                for t in self.numpy_trees
            ],
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.miss_threshold = float(data.get("miss_threshold", self.miss_threshold))
        self.numpy_trees = [
            SimpleTree(t["feature_idx"], t["threshold"], t["left_val"], t["right_val"])
            for t in data.get("trees", [])
        ]
