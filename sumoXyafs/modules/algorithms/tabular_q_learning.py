"""Algorithm 7: Tabular Q-Learning for RSU offloading and migration."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import DecisionFeedback, OffloadingState


class TabularQLearningSelector(RSUSelector):
    """Tabular Q-Learning reinforcement learning algorithm.

    Uses a coarse state discretization (load level, distance bucket, deadline slack)
    and a small 27 x 4 Q-table mapping to offloading meta-actions:
    0: Stay local (observed RSU)
    1: Sticky (keep current execution RSU)
    2: Offload to nearest available RSU
    3: Offload to least loaded RSU
    """

    name: str = "q_learning"

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        # 27 states (3 x 3 x 3), 4 actions
        self.Q = np.zeros((27, 4), dtype=float)
        # Optimistic initialization for actions 0 and 1
        self.Q[:, 0] = 0.5
        self.Q[:, 1] = 0.5
        self._last_state_action: dict[str, tuple[int, int]] = {}

    def _discretize_state(self, state: OffloadingState) -> int:
        curr_rsu = state.execution_rsu or state.observed_rsu
        q = state.queue_depths.get(curr_rsu, 0)
        # Load bucket: 0 (low), 1 (med), 2 (high)
        load_bin = 0 if q < 5 else (1 if q <= 15 else 2)

        dist = state.rsu_distances.get(curr_rsu, 500.0)
        # Distance bucket: 0 (near), 1 (mid), 2 (far)
        dist_bin = 0 if dist < 300.0 else (1 if dist <= 800.0 else 2)

        slack = state.deadline_slack_steps
        # Slack bucket: 0 (tight), 1 (moderate), 2 (ample)
        slack_bin = 0 if slack <= 0 else (1 if slack <= 3 else 2)

        return load_bin * 9 + dist_bin * 3 + slack_bin

    def _map_action_to_rsu(self, action: int, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        if action == 0:  # Stay local
            return state.observed_rsu

        if action == 1:  # Sticky with current exec RSU
            if state.execution_rsu in candidates:
                return state.execution_rsu
            return state.observed_rsu

        if action == 2:  # Nearest available RSU
            best = None
            best_dist = float("inf")
            for r in candidates:
                q = state.queue_depths.get(r, 0)
                cap = state.rsu_capacities.get(r, 20)
                if q < cap:
                    d = state.rsu_distances.get(r, float("inf"))
                    if d < best_dist:
                        best_dist = d
                        best = r
            return best or state.observed_rsu

        # Action 3: Least loaded RSU
        return min(candidates, key=lambda r: state.queue_depths.get(r, 0))

    def choose_rsu(self, state: OffloadingState) -> str:
        s_idx = self._discretize_state(state)

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = int(np.random.randint(0, 4))
        else:
            action = int(np.argmax(self.Q[s_idx]))

        self._last_state_action[state.vehicle_id] = (s_idx, action)
        return self._map_action_to_rsu(action, state)

    def update(self, feedback: DecisionFeedback) -> None:
        last = self._last_state_action.get(feedback.vehicle_id)
        if last is None:
            return

        s_idx, action = last
        next_s_idx = self._discretize_state(feedback.state)

        # Reward: +1.0 for deadline met, -1.5 for miss, slight penalty for latency
        reward = 1.0 if feedback.deadline_met else -1.5
        reward -= 0.1 * min(5.0, feedback.latency_s)

        # Q-learning update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
        best_next_q = float(np.max(self.Q[next_s_idx]))
        td_target = reward + self.gamma * best_next_q
        self.Q[s_idx, action] += self.alpha * (td_target - self.Q[s_idx, action])

        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def save(self, file_path: str | Path) -> None:
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "Q": self.Q.tolist(),
        }
        Path(file_path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.alpha = float(data.get("alpha", self.alpha))
        self.gamma = float(data.get("gamma", self.gamma))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.Q = np.array(data.get("Q", self.Q), dtype=float)
