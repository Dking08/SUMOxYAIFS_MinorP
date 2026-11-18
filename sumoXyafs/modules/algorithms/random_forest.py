"""Algorithm 4: Random Forest Classifier for deadline-miss filtering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.algorithms.decision_tree import DecisionTreeNode, fit_numpy_tree, sklearn_tree_to_node
from modules.models import OffloadingState

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class SimpleTree:
    """Backward-compatible wrapper for single-stump evaluations."""

    def __init__(self, feature_idx: int = 0, threshold: float = 0.5, left_val: float = 0.0, right_val: float = 1.0) -> None:
        self.node = DecisionTreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=DecisionTreeNode(value=left_val, is_leaf=True),
            right=DecisionTreeNode(value=right_val, is_leaf=True),
            value=(left_val + right_val) / 2.0,
            is_leaf=False,
        )

    def predict(self, x: np.ndarray) -> float:
        return self.node.predict(x)


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
        self.sklearn_model: Optional[object] = None
        self.numpy_trees: list[DecisionTreeNode] = []
        self._init_prior_trees()

    def _init_prior_trees(self) -> None:
        # Default physical heuristics encoded as randomized tree ensemble:
        # Features: [0: norm_dist, 1: norm_q, 2: norm_speed, 3: norm_slack, 4: is_observed]
        stump_defs = [
            (1, 0.85, 0.1, 0.9),  # high queue -> miss
            (3, -0.1, 0.85, 0.15),  # tight slack -> miss
            (0, 0.75, 0.2, 0.7),  # far distance -> miss
            (1, 0.60, 0.15, 0.65),
            (3, 0.20, 0.6, 0.1),
        ]
        self.numpy_trees = [
            DecisionTreeNode(
                feature_idx=f,
                threshold=th,
                left=DecisionTreeNode(value=lv, is_leaf=True),
                right=DecisionTreeNode(value=rv, is_leaf=True),
                is_leaf=False,
            )
            for f, th, lv, rv in stump_defs
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
            try:
                proba = self.sklearn_model.predict_proba(features.reshape(1, -1))[0]
                return float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                pass

        if not self.numpy_trees:
            return 0.5
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
            return min(candidates, key=lambda r: rsu_miss_probs[r])

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
            miss = 1 - int(row["deadline_met"])

            X_list.append([dist, norm_q, speed, slack, is_obs])
            y_list.append(miss)

        if len(X_list) < 5:
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if HAS_SKLEARN:
            try:
                model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=5, random_state=42)
                model.fit(X, y)
                self.sklearn_model = model
                self.numpy_trees = [sklearn_tree_to_node(est.tree_) for est in model.estimators_]
                return
            except Exception:
                pass

        # Pure-NumPy bootstrap bagged forest
        np.random.seed(42)
        trees: list[DecisionTreeNode] = []
        n_samples = len(X)
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            trees.append(fit_numpy_tree(X[idx], y[idx], max_depth=4))
        self.numpy_trees = trees

    def save(self, file_path: str | Path) -> None:
        data = {
            "miss_threshold": self.miss_threshold,
            "n_estimators": len(self.numpy_trees),
            "trees": [t.to_dict() for t in self.numpy_trees],
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.miss_threshold = float(data.get("miss_threshold", self.miss_threshold))
        raw_trees = data.get("trees", [])
        self.numpy_trees = [DecisionTreeNode.from_dict(t) for t in raw_trees]
