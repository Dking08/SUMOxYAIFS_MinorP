"""Per-vehicle task lifecycle tracking with GFLOPs computational demands and hardware profiles."""
from __future__ import annotations

import itertools
import math
from typing import Optional

from modules.models import TaskState, Telemetry


class TaskManager:
    def __init__(self, task_deadline_s: float = 2.0, compute_time_s: float = 2.0):
        self.task_deadline_s = max(0.2, float(task_deadline_s))
        self.compute_time_s = max(0.1, float(compute_time_s))
        self._task_counter = itertools.count(1)
        self._active: dict[str, TaskState] = {}
        self.manual_overrides: dict[str, str] = {}
        self.queue_overrides: dict[str, int] = {}

    def set_manual_override(self, vehicle_id: str, target_rsu: str):
        self.manual_overrides[vehicle_id] = target_rsu

    def set_rsu_queue_override(self, rsu_id: str, depth: int):
        self.queue_overrides[rsu_id] = max(0, int(depth))

    def get(self, vehicle_id: str) -> Optional[TaskState]:
        return self._active.get(vehicle_id)

    def upsert_task(
        self,
        telemetry: Telemetry,
        owner_rsu: str,
        category: str = "traffic",
    ) -> TaskState:
        state = self._active.get(telemetry.vehicle_id)

        # Create a new task whenever the previous one is absent or completed.
        if state is None or state.remaining_gflops <= 0.0 or state.remaining_compute_s <= 0.0:
            task_num = next(self._task_counter)

            # Assign task computational parameters based on task category
            cat = str(category).lower().strip()
            if cat == "safety":
                total_gflops = 0.5
                p_in, p_out = 0.2, 0.05
                ddl_s = 0.5
            elif cat == "global":
                total_gflops = 50.0
                p_in, p_out = 10.0, 1.0
                ddl_s = max(5.0, self.task_deadline_s * 3.0)
            else:  # "traffic" or default
                total_gflops = max(0.5, self.compute_time_s * 1.0)
                p_in, p_out = 1.0, 0.2
                ddl_s = self.task_deadline_s

            deadline_step = telemetry.step + int(math.ceil(ddl_s))
            state = TaskState(
                task_id=f"task_{task_num}",
                vehicle_id=telemetry.vehicle_id,
                owner_rsu=owner_rsu,
                execution_rsu=owner_rsu,
                created_step=telemetry.step,
                deadline_step=deadline_step,
                remaining_compute_s=self.compute_time_s,
                remaining_gflops=total_gflops,
                total_gflops=total_gflops,
                payload_in_mbits=p_in,
                payload_out_mbits=p_out,
                task_category=cat,
                execution_tier="V2I",
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
        node_gflops: float = 3200.0,
    ) -> int:
        """Estimates completion step using queue depth, task compute time, and network latency."""
        queue_delay = max(0.0, float(queue_depth))
        compute_delay = max(0.1, float(state.remaining_compute_s))
        network_delay = max(0.0, float(network_latency_s))
        return now_step + int(math.ceil(queue_delay + compute_delay + network_delay))

    def migrate(self, state: TaskState, new_owner_rsu: str, new_execution_rsu: str, tier: str = "FOG_MIGRATE") -> None:
        state.owner_rsu = new_owner_rsu
        state.execution_rsu = new_execution_rsu
        state.execution_tier = tier
        state.migrations += 1

    def set_execution_rsu(self, state: TaskState, execution_rsu: str, tier: str = "V2I") -> None:
        state.execution_rsu = execution_rsu
        state.execution_tier = tier

    def tick_all(
        self,
        rsu_registry=None,
        step_size_s: float = 1.0,
        current_step: Optional[int] = None,
    ) -> list[TaskState]:
        completed: list[TaskState] = []
        if not self._active:
            return completed

        remove_ids: list[str] = []
        for vid, state in self._active.items():
            # Determine execution node processing speed
            node_gflops = 3200.0  # Default Siemens RS22X speed
            if rsu_registry is not None and state.execution_rsu in rsu_registry._by_id:
                node_gflops = rsu_registry.get(state.execution_rsu).spec.gflops
            elif state.execution_tier == "V2V":
                node_gflops = 472.0  # Jetson Nano OBU
            elif state.execution_tier == "CLOUD":
                node_gflops = 1000000.0  # Cloud Data Center

            gflops_processed = node_gflops * float(step_size_s)
            state.remaining_gflops = max(0.0, state.remaining_gflops - gflops_processed)
            state.remaining_compute_s = max(0.0, state.remaining_compute_s - float(step_size_s))

            if state.total_gflops > 0:
                state.progress = min(1.0, 1.0 - (state.remaining_gflops / state.total_gflops))

            if state.remaining_gflops <= 0.0 or state.remaining_compute_s <= 0.0:
                state.progress = 1.0
                finish_step = (
                    current_step
                    if current_step is not None
                    else state.created_step + max(1, int(math.ceil(state.total_gflops / max(1.0, node_gflops * step_size_s))))
                )
                state.completed_step = finish_step
                state.actual_latency_s = max(0.05, float(finish_step - state.created_step))
                state.deadline_met = finish_step <= state.deadline_step
                completed.append(state)
                remove_ids.append(vid)

        for vid in remove_ids:
            self._active.pop(vid, None)
        return completed

    def prune_missing(self, active_vehicle_ids: set[str], current_step: Optional[int] = None) -> list[TaskState]:
        stale = [vid for vid in self._active if vid not in active_vehicle_ids]
        dropped: list[TaskState] = []
        for vid in stale:
            task = self._active.pop(vid, None)
            if task is not None:
                drop_step = current_step if current_step is not None else task.deadline_step + 1
                task.completed_step = drop_step
                task.actual_latency_s = float(drop_step - task.created_step)
                task.deadline_met = False
                dropped.append(task)
        return dropped

    def active_queue_depths(self) -> dict[str, int]:
        queue_depths: dict[str, int] = {}
        for rsu_id, q_val in self.queue_overrides.items():
            queue_depths[rsu_id] = max(0, int(q_val))
        for state in self._active.values():
            rsu = state.execution_rsu
            queue_depths[rsu] = queue_depths.get(rsu, 0) + 1
        return queue_depths
