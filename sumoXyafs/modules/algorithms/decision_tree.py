"""Algorithm 6: Decision Tree / Trigger Rule for Migrate vs. Stay decisions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState

try:
    from sklearn.tree import DecisionTreeClassifier  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class DecisionTreeNode:
    """Portable decision tree node supporting JSON serialization and pure-NumPy inference."""

    def __init__(
        self,
        feature_idx: int = -1,
        threshold: float = 0.0,
        left: Optional[DecisionTreeNode] = None,
        right: Optional[DecisionTreeNode] = None,
        value: float = 0.0,
        is_leaf: bool = True,
    ) -> None:
        self.feature_idx = int(feature_idx)
        self.threshold = float(threshold)
        self.left = left
        self.right = right
        self.value = float(value)
        self.is_leaf = bool(is_leaf)

    def predict(self, x: np.ndarray) -> float:
        if self.is_leaf:
            return self.value
        val = float(x[self.feature_idx])
        if val <= self.threshold:
            return self.left.predict(x) if self.left is not None else self.value
        return self.right.predict(x) if self.right is not None else self.value

    def to_dict(self) -> dict:
        if self.is_leaf:
            return {"is_leaf": True, "value": self.value}
        return {
            "is_leaf": False,
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "value": self.value,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DecisionTreeNode:
        if not d:
            return cls(value=0.5, is_leaf=True)
        if "left_val" in d and "right_val" in d:
            f = int(d.get("feature_idx", 0))
            th = float(d.get("threshold", 0.5))
            lv = float(d.get("left_val", 0.0))
            rv = float(d.get("right_val", 1.0))
            return cls(
                feature_idx=f,
                threshold=th,
                left=cls(value=lv, is_leaf=True),
                right=cls(value=rv, is_leaf=True),
                value=(lv + rv) / 2.0,
                is_leaf=False,
            )
        if d.get("is_leaf", False):
            return cls(value=float(d.get("value", 0.0)), is_leaf=True)
        left = cls.from_dict(d["left"]) if d.get("left") else None
        right = cls.from_dict(d["right"]) if d.get("right") else None
        return cls(
            feature_idx=int(d.get("feature_idx", -1)),
            threshold=float(d.get("threshold", 0.0)),
            left=left,
            right=right,
            value=float(d.get("value", 0.0)),
            is_leaf=False,
        )


def sklearn_tree_to_node(tree, node_id: int = 0) -> DecisionTreeNode:
    left_child = int(tree.children_left[node_id])
    right_child = int(tree.children_right[node_id])
    counts = tree.value[node_id][0]
    total = float(np.sum(counts))
    prob1 = float(counts[1] / total) if len(counts) > 1 and total > 0 else float(np.argmax(counts))
    if left_child == -1:
        return DecisionTreeNode(value=prob1, is_leaf=True)
    return DecisionTreeNode(
        feature_idx=int(tree.feature[node_id]),
        threshold=float(tree.threshold[node_id]),
        left=sklearn_tree_to_node(tree, left_child),
        right=sklearn_tree_to_node(tree, right_child),
        value=prob1,
        is_leaf=False,
    )


def fit_numpy_tree(X: np.ndarray, y: np.ndarray, max_depth: int = 4, min_samples: int = 4) -> DecisionTreeNode:
    n_samples, n_features = X.shape
    if n_samples < min_samples or max_depth <= 0 or np.all(y == y[0]):
        return DecisionTreeNode(value=float(np.mean(y)), is_leaf=True)

    best_gini = float("inf")
    best_feat = -1
    best_th = 0.0
    best_left = None

    p0 = float(np.mean(y))
    current_gini = 2.0 * p0 * (1.0 - p0)

    for f in range(n_features):
        vals = np.unique(X[:, f])
        if len(vals) <= 1:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0 if len(vals) <= 10 else np.percentile(vals, [25, 50, 75])
        for th in thresholds:
            l_mask = X[:, f] <= th
            nl = int(np.sum(l_mask))
            nr = n_samples - nl
            if nl < 2 or nr < 2:
                continue
            pl = float(np.mean(y[l_mask]))
            pr = float(np.mean(y[~l_mask]))
            gini = (nl * 2.0 * pl * (1.0 - pl) + nr * 2.0 * pr * (1.0 - pr)) / n_samples
            if gini < best_gini:
                best_gini = gini
                best_feat = f
                best_th = float(th)
                best_left = l_mask

    if best_feat == -1 or best_gini >= current_gini:
        return DecisionTreeNode(value=float(np.mean(y)), is_leaf=True)

    left_node = fit_numpy_tree(X[best_left], y[best_left], max_depth=max_depth - 1, min_samples=min_samples)
    right_node = fit_numpy_tree(X[~best_left], y[~best_left], max_depth=max_depth - 1, min_samples=min_samples)
    return DecisionTreeNode(
        feature_idx=best_feat,
        threshold=best_th,
        left=left_node,
        right=right_node,
        value=float(np.mean(y)),
        is_leaf=False,
    )


class DecisionTreeSelector(RSUSelector):
    """Supervised Decision Tree classifier deciding 'Migrate or Stay'.

    Labels past decisions by whether staying caused a deadline miss;
    the fitted tree becomes the trigger rule for proactive migration.
    """

    name: str = "decision_tree"

    def __init__(self, max_depth: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_depth = int(max_depth)
        self.root: Optional[DecisionTreeNode] = None
        self.sklearn_tree: Optional[object] = None
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
        if self.root is not None:
            pred = self.root.predict(feats)
            return bool(pred >= 0.5)

        if self.sklearn_tree is not None and HAS_SKLEARN:
            pred = self.sklearn_tree.predict(feats.reshape(1, -1))[0]
            return bool(pred == 1)

        # Interpretable threshold fallback
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
            is_stay = action in {"keep_local", "sticky", "nominal", ""}
            if not is_stay and "action" in row and row.get("action"):
                continue

            q = float(row.get("queue_depth_action", 0)) / 20.0
            dist = float(row.get("norm_distance_to_action", 0.5))
            speed = min(1.0, float(row.get("speed", 0.0)) / 30.0)
            slack = max(-1.0, min(1.0, float(row.get("deadline_slack_steps", 0)) / 10.0))

            target = 1 if int(row["deadline_met"]) == 0 else 0
            X_list.append([q, dist, speed, slack])
            y_list.append(target)

        if len(X_list) < 5:
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        # Always calibrate threshold rules as interpretable bounds
        miss_idx = y == 1
        if np.any(miss_idx):
            self.rules["queue_thresh"] = float(np.percentile(X[miss_idx, 0], 25))
            self.rules["slack_thresh"] = float(np.percentile(X[miss_idx, 3], 75))

        if HAS_SKLEARN:
            try:
                tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
                tree.fit(X, y)
                self.sklearn_tree = tree
                self.root = sklearn_tree_to_node(tree.tree_)
                return
            except Exception:
                pass

        # Pure-NumPy tree training
        self.root = fit_numpy_tree(X, y, max_depth=self.max_depth)

    def save(self, file_path: str | Path) -> None:
        data = {
            "rules": self.rules,
            "max_depth": self.max_depth,
            "tree": self.root.to_dict() if self.root is not None else None,
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.rules = data.get("rules", self.rules)
        self.max_depth = int(data.get("max_depth", self.max_depth))
        tree_dict = data.get("tree")
        if tree_dict:
            self.root = DecisionTreeNode.from_dict(tree_dict)
        else:
            self.root = None
