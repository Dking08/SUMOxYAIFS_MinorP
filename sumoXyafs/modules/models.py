"""Core models used by the SUMO <-> RSU <-> fog bridge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


Position = Tuple[float, float]


@dataclass
class Telemetry:
    vehicle_id: str
    position: Position
    speed: float
    step: int
    observed_rsu: str = ""


@dataclass
class RSUNode:
    rsu_id: str
    node_id: int
    x: float
    y: float
    capacity: int = 20
    neighbors: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    task_id: str
    vehicle_id: str
    owner_rsu: str
    execution_rsu: str
    created_step: int
    deadline_step: int
    remaining_compute_s: float
    progress: float = 0.0
    last_position: Position = (0.0, 0.0)
    last_speed: float = 0.0
    migrations: int = 0


@dataclass
class SchedulingDecision:
    vehicle_id: str
    observed_rsu: str
    owner_rsu: str
    execution_rsu: str
    action: str
    reason: str
    estimated_finish_step: int
    deadline_step: int
    estimated_network_latency_s: float = 0.0


@dataclass
class ControlCommand:
    vehicle_id: str
    action: str
    speed: float
    step: int
    position: Position
    source_rsu: str
    reroute: bool = False
    reason: str = ""
    task_id: Optional[str] = None

    def to_packet(self) -> dict:
        return {
            "id": self.vehicle_id,
            "action": self.action,
            "speed": self.speed,
            "time": self.step,
            "position": self.position,
            "source_rsu": self.source_rsu,
            "reroute": self.reroute,
            "reason": self.reason,
            "task_id": self.task_id,
        }


@dataclass
class OffloadingState:
    vehicle_id: str
    position: Position
    speed: float
    step: int
    observed_rsu: str
    owner_rsu: str
    execution_rsu: str
    queue_depths: dict[str, int] = field(default_factory=dict)
    rsu_distances: dict[str, float] = field(default_factory=dict)
    rsu_capacities: dict[str, int] = field(default_factory=dict)
    task_id: str = ""
    remaining_compute_s: float = 2.0
    deadline_step: int = 0
    deadline_slack_steps: int = 0
    heading_deg: float = 0.0
    candidate_rsus: list[str] = field(default_factory=list)

    def get_candidate_features(self, rsu_id: str) -> dict[str, float]:
        dist = self.rsu_distances.get(rsu_id, 1000.0)
        q = float(self.queue_depths.get(rsu_id, 0))
        cap = float(self.rsu_capacities.get(rsu_id, 20))
        cap = max(1.0, cap)
        return {
            "norm_distance": min(1.0, dist / 1500.0),
            "norm_queue": min(1.0, q / cap),
            "norm_speed": min(1.0, self.speed / 30.0),
            "norm_slack": max(-1.0, min(1.0, self.deadline_slack_steps / 10.0)),
            "is_observed": 1.0 if rsu_id == self.observed_rsu else 0.0,
            "is_current_exec": 1.0 if rsu_id == self.execution_rsu else 0.0,
        }

    def to_feature_vector(self, rsu_id: str) -> list[float]:
        feats = self.get_candidate_features(rsu_id)
        return [
            feats["norm_distance"],
            feats["norm_queue"],
            feats["norm_speed"],
            feats["norm_slack"],
            feats["is_observed"],
            feats["is_current_exec"],
            1.0,  # bias
        ]

    def to_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "position_x": self.position[0],
            "position_y": self.position[1],
            "speed": self.speed,
            "step": self.step,
            "observed_rsu": self.observed_rsu,
            "owner_rsu": self.owner_rsu,
            "execution_rsu": self.execution_rsu,
            "task_id": self.task_id,
            "remaining_compute_s": self.remaining_compute_s,
            "deadline_step": self.deadline_step,
            "deadline_slack_steps": self.deadline_slack_steps,
            "heading_deg": self.heading_deg,
        }


@dataclass
class DecisionFeedback:
    state: OffloadingState
    action_rsu: str
    latency_s: float
    deadline_met: bool
    reward: float
    step: int
    vehicle_id: str

    def to_dict(self) -> dict:
        d = self.state.to_dict()
        d.update(
            {
                "action_rsu": self.action_rsu,
                "latency_s": self.latency_s,
                "deadline_met": int(self.deadline_met),
                "reward": self.reward,
                "action_step": self.step,
            }
        )
        return d

