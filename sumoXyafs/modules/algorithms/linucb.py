"""Algorithm 1: LinUCB Contextual Bandit for RSU offloading."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import DecisionFeedback, OffloadingState


class LinUCBSelector(RSUSelector):
    """LinUCB online contextual bandit.

    Replaces greedy score with a ridge-regression reward estimate
    plus an upper confidence bound (UCB) exploration bonus.
    """

    name: str = "linucb"

    def __init__(self, alpha: float = 0.5, reg_lambda: float = 1.0, d: int = 7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.reg_lambda = float(reg_lambda)
        self.d = int(d)
        # Arm matrices: A_a (d x d), b_a (d x 1)
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}

    def _ensure_arm(self, rsu_id: str) -> None:
        if rsu_id not in self.A:
            self.A[rsu_id] = np.eye(self.d, dtype=float) * self.reg_lambda
            self.b[rsu_id] = np.zeros((self.d, 1), dtype=float)

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        best_score = -float("inf")
        best_rsu = candidates[0]

        for rsu_id in candidates:
            self._ensure_arm(rsu_id)
            x = np.array(state.to_feature_vector(rsu_id), dtype=float).reshape(-1, 1)

            # A_inv = np.linalg.pinv(self.A[rsu_id])
            A_inv = np.linalg.solve(self.A[rsu_id], np.eye(self.d))
            theta = A_inv @ self.b[rsu_id]
            expected_reward = float((theta.T @ x).item())
            variance = float((x.T @ A_inv @ x).item())
            exploration_bonus = self.alpha * np.sqrt(max(0.0, variance))
            score = expected_reward + exploration_bonus

            # Penalize overloaded RSUs
            q = state.queue_depths.get(rsu_id, 0)
            cap = state.rsu_capacities.get(rsu_id, 20)
            if q >= cap:
                score -= 5.0

            if score > best_score:
                best_score = score
                best_rsu = rsu_id

        return best_rsu

    def update(self, feedback: DecisionFeedback) -> None:
        rsu_id = feedback.action_rsu
        self._ensure_arm(rsu_id)
        x = np.array(feedback.state.to_feature_vector(rsu_id), dtype=float).reshape(-1, 1)
        # Reward: 1.0 for deadline met, scaled negative penalty otherwise
        reward = float(feedback.reward)
        self.A[rsu_id] += x @ x.T
        self.b[rsu_id] += reward * x

    def save(self, file_path: str | Path) -> None:
        data = {
            "alpha": self.alpha,
            "reg_lambda": self.reg_lambda,
            "d": self.d,
            "arms": {
                rsu_id: {
                    "A": self.A[rsu_id].tolist(),
                    "b": self.b[rsu_id].tolist(),
                }
                for rsu_id in self.A
            },
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.alpha = float(data.get("alpha", self.alpha))
        self.reg_lambda = float(data.get("reg_lambda", self.reg_lambda))
        self.d = int(data.get("d", self.d))
        self.A = {k: np.array(v["A"], dtype=float) for k, v in data.get("arms", {}).items()}
        self.b = {k: np.array(v["b"], dtype=float) for k, v in data.get("arms", {}).items()}
