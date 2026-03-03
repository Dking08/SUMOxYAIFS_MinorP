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
