"""Algorithm 2: Thompson Sampling Bandit for RSU offloading."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import DecisionFeedback, OffloadingState


class ThompsonSamplingSelector(RSUSelector):
    """Thompson Sampling online bandit.

    Maintains a Beta posterior over deadline-met success probability per RSU,
    samples from it at decision time, and penalizes heavy queue load and excessive distance.
    """

    name: str = "thompson"

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
        dist_penalty_weight: float = 0.25,
        load_penalty_weight: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prior_alpha = max(0.1, float(prior_alpha))
        self.prior_beta = max(0.1, float(prior_beta))
        self.dist_penalty_weight = float(dist_penalty_weight)
        self.load_penalty_weight = float(load_penalty_weight)
        # Success and failure counts per RSU
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def _ensure_arm(self, rsu_id: str) -> None:
        if rsu_id not in self.alpha:
            self.alpha[rsu_id] = self.prior_alpha
            self.beta[rsu_id] = self.prior_beta

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        best_score = -float("inf")
        best_rsu = candidates[0]

        for rsu_id in candidates:
            self._ensure_arm(rsu_id)
            # Sample success probability from Beta posterior
            sampled_theta = np.random.beta(self.alpha[rsu_id], self.beta[rsu_id])

            dist = state.rsu_distances.get(rsu_id, 1000.0)
            norm_dist = min(1.0, dist / 1500.0)

            q = state.queue_depths.get(rsu_id, 0)
            cap = max(1, state.rsu_capacities.get(rsu_id, 20))
            norm_q = min(1.0, q / cap)

            # Combined score: sampled posterior reward minus contextual friction
            score = sampled_theta - (self.dist_penalty_weight * norm_dist) - (self.load_penalty_weight * norm_q)

            # Hard penalty if strictly overloaded
            if q >= cap:
                score -= 3.0

            if score > best_score:
                best_score = score
                best_rsu = rsu_id

        return best_rsu

    def update(self, feedback: DecisionFeedback) -> None:
        rsu_id = feedback.action_rsu
        self._ensure_arm(rsu_id)
        if feedback.deadline_met:
            self.alpha[rsu_id] += 1.0
        else:
            self.beta[rsu_id] += 1.0

    def save(self, file_path: str | Path) -> None:
        data = {
            "alpha": self.alpha,
            "beta": self.beta,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.alpha = {k: float(v) for k, v in data.get("alpha", {}).items()}
        self.beta = {k: float(v) for k, v in data.get("beta", {}).items()}
