"""Base interface and dataset logging for RSU selection algorithms."""
from __future__ import annotations

import abc
import csv
from pathlib import Path
from typing import Any, Optional

from modules.models import DecisionFeedback, OffloadingState


class RSUSelector(abc.ABC):
    """Abstract base class for all RSU offloading and scheduling algorithms."""

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abc.abstractmethod
    def choose_rsu(self, state: OffloadingState) -> str:
        """Choose target RSU id given current offloading state.

        Args:
            state: OffloadingState snapshot containing vehicle telemetry,
                   RSU distances, current queues, and task deadlines.

        Returns:
            The selected RSU id (e.g. 'rsu_1').
        """
        raise NotImplementedError

    def update(self, feedback: DecisionFeedback) -> None:
        """Online learning update hook.

        Used by online learners (LinUCB, Thompson Sampling, Tabular Q-learning).
        No-op by default for static/supervised algorithms.
        """
        pass

    def train_offline(self, records: list[dict]) -> None:
        """Offline training hook using historical decision records.

        Used by supervised / statistical algorithms (Ridge, Random Forest,
        Markov Chain, Decision Tree).
        """
        pass

    def save(self, file_path: str | Path) -> None:
        """Serialize trained weights or policy parameters."""
        pass

    def load(self, file_path: str | Path) -> None:
        """Load trained weights or policy parameters."""
        pass


class DatasetLogger:
    """Logs (state, action, latency, deadline_met, reward) for offline model training."""

    def __init__(self, output_path: str | Path) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._fieldnames = [
            "step",
            "vehicle_id",
            "position_x",
            "position_y",
            "speed",
            "observed_rsu",
            "owner_rsu",
            "execution_rsu",
            "action_rsu",
            "latency_s",
            "deadline_met",
            "reward",
            "remaining_compute_s",
            "deadline_step",
            "deadline_slack_steps",
            "heading_deg",
            "norm_distance_to_action",
            "queue_depth_action",
        ]
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
                writer.writeheader()

    def log(self, feedback: DecisionFeedback) -> None:
        st = feedback.state
        dist = st.rsu_distances.get(feedback.action_rsu, 0.0)
        q = st.queue_depths.get(feedback.action_rsu, 0)
        row = {
            "step": feedback.step,
            "vehicle_id": feedback.vehicle_id,
            "position_x": st.position[0],
            "position_y": st.position[1],
            "speed": st.speed,
            "observed_rsu": st.observed_rsu,
            "owner_rsu": st.owner_rsu,
            "execution_rsu": st.execution_rsu,
            "action_rsu": feedback.action_rsu,
            "latency_s": feedback.latency_s,
            "deadline_met": int(feedback.deadline_met),
            "reward": feedback.reward,
            "remaining_compute_s": st.remaining_compute_s,
            "deadline_step": st.deadline_step,
            "deadline_slack_steps": st.deadline_slack_steps,
            "heading_deg": st.heading_deg,
            "norm_distance_to_action": min(1.0, dist / 1500.0),
            "queue_depth_action": q,
        }
        self._buffer.append(row)
        if len(self._buffer) >= 50:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            for row in self._buffer:
                writer.writerow(row)
        self._buffer.clear()
