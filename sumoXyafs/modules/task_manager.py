"""Per-vehicle task lifecycle tracking for fog processing."""
from __future__ import annotations

import itertools
import math
from typing import Optional

from modules.models import TaskState, Telemetry


class TaskManager:
    def __init__(self, task_deadline_s: float = 2.0, compute_time_s: float = 2.0):
        self.task_deadline_s = max(1.0, float(task_deadline_s))
        self.compute_time_s = max(0.1, float(compute_time_s))
        self._task_counter = itertools.count(1)
        self._active: dict[str, TaskState] = {}

    def get(self, vehicle_id: str) -> Optional[TaskState]:
        return self._active.get(vehicle_id)

    def upsert_task(self, telemetry: Telemetry, owner_rsu: str) -> TaskState:
        state = self._active.get(telemetry.vehicle_id)

        # Create a new task whenever the previous one is absent or finished.
        if state is None or state.remaining_compute_s <= 0.0:
            task_num = next(self._task_counter)
            deadline_step = telemetry.step + int(math.ceil(self.task_deadline_s))
            state = TaskState(
                task_id=f"task_{task_num}",
                vehicle_id=telemetry.vehicle_id,
                owner_rsu=owner_rsu,
                execution_rsu=owner_rsu,
                created_step=telemetry.step,
                deadline_step=deadline_step,
                remaining_compute_s=self.compute_time_s,
                progress=0.0,
                last_position=telemetry.position,
                last_speed=telemetry.speed,
                migrations=0,
            )
            self._active[telemetry.vehicle_id] = state
        else:
            state.last_position = telemetry.position
            state.last_speed = telemetry.speed

        return state

    def estimate_finish_step(
        self,
        state: TaskState,
        queue_depth: int,
        now_step: int,
        network_latency_s: float = 0.0,
    ) -> int:
        queue_delay = max(0.0, float(queue_depth))
        compute_delay = max(0.0, float(state.remaining_compute_s))
        network_delay = max(0.0, float(network_latency_s))
        return now_step + int(math.ceil(queue_delay + compute_delay + network_delay))

    def migrate(self, state: TaskState, new_owner_rsu: str, new_execution_rsu: str) -> None:
        state.owner_rsu = new_owner_rsu
        state.execution_rsu = new_execution_rsu
        state.migrations += 1

    def set_execution_rsu(self, state: TaskState, execution_rsu: str) -> None:
        state.execution_rsu = execution_rsu

    def tick_all(self, step_size_s: float = 1.0) -> list[TaskState]:
        completed: list[TaskState] = []
        if not self._active:
            return completed

        # Move all active tasks forward by one simulation step.
        remove_ids: list[str] = []
        for vid, state in self._active.items():
            before = state.remaining_compute_s
            state.remaining_compute_s = max(0.0, state.remaining_compute_s - float(step_size_s))
            if self.compute_time_s > 0:
                state.progress = min(1.0, 1.0 - (state.remaining_compute_s / self.compute_time_s))
            if before > 0.0 and state.remaining_compute_s <= 0.0:
                completed.append(state)
                remove_ids.append(vid)

        for vid in remove_ids:
            self._active.pop(vid, None)
        return completed

    def prune_missing(self, active_vehicle_ids: set[str]) -> None:
        stale = [vid for vid in self._active if vid not in active_vehicle_ids]
        for vid in stale:
            self._active.pop(vid, None)

    def active_queue_depths(self) -> dict[str, int]:
        queue_depths: dict[str, int] = {}
        for state in self._active.values():
            queue_depths[state.execution_rsu] = queue_depths.get(state.execution_rsu, 0) + 1
        return queue_depths
