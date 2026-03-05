"""Scheduling policy for RSU ownership, handover, and overload offload."""
from __future__ import annotations

from modules.models import SchedulingDecision, Telemetry
from modules.rsu_registry import RSURegistry
from modules.task_manager import TaskManager


class Scheduler:
    def __init__(
        self,
        rsu_registry: RSURegistry,
        task_manager: TaskManager,
        overload_threshold: int = 20,
        distance_latency_base_s: float = 0.0,
        distance_latency_max_s: float = 0.0,
        distance_latency_norm_m: float = 500.0,
    ):
        self.rsu_registry = rsu_registry
        self.task_manager = task_manager
        self.overload_threshold = max(1, int(overload_threshold))
        self.distance_latency_base_s = max(0.0, float(distance_latency_base_s))
        self.distance_latency_max_s = max(self.distance_latency_base_s, float(distance_latency_max_s))
        self.distance_latency_norm_m = max(1.0, float(distance_latency_norm_m))

    def schedule(
        self,
        telemetry: Telemetry,
        queue_depths: dict[str, int],
    ) -> SchedulingDecision:
        state = self.task_manager.upsert_task(telemetry, telemetry.observed_rsu)

        action = "keep_local"
        reason = "same_region"

        # Apply deadline-aware sticky handover when observed RSU changes.
        if telemetry.observed_rsu != state.owner_rsu:
            current_queue = queue_depths.get(state.execution_rsu, 0)
            sticky_latency = self._distance_latency_s(telemetry.position, state.execution_rsu)
            estimated_finish = self.task_manager.estimate_finish_step(
                state,
                current_queue,
                telemetry.step,
                network_latency_s=sticky_latency,
            )
            if estimated_finish <= state.deadline_step:
                action = "sticky"
                reason = "can_finish_before_deadline"
            else:
                self.task_manager.migrate(state, telemetry.observed_rsu, telemetry.observed_rsu)
                action = "migrate"
                reason = "deadline_risk"

        # If the selected execution RSU is overloaded, offload to nearest available RSU.
        execution_rsu = state.execution_rsu
        if self.rsu_registry.is_overloaded(execution_rsu, queue_depths, self.overload_threshold):
            alt = self.rsu_registry.nearest_available_rsu(
                telemetry.position,
                queue_depths,
                self.overload_threshold,
                exclude={execution_rsu},
            )
            if alt is not None:
                self.task_manager.set_execution_rsu(state, alt.rsu_id)
                execution_rsu = alt.rsu_id
                if action == "keep_local":
                    action = "offload"
                    reason = "overloaded_exec_rsu"
                else:
                    reason = f"{reason}+offloaded"

        queue_depths[execution_rsu] = queue_depths.get(execution_rsu, 0) + 1
        exec_latency = self._distance_latency_s(telemetry.position, execution_rsu)
        estimated_finish = self.task_manager.estimate_finish_step(
            state,
            queue_depths.get(execution_rsu, 0),
            telemetry.step,
            network_latency_s=exec_latency,
        )

        return SchedulingDecision(
            vehicle_id=telemetry.vehicle_id,
            observed_rsu=telemetry.observed_rsu,
            owner_rsu=state.owner_rsu,
            execution_rsu=execution_rsu,
            action=action,
            reason=reason,
            estimated_finish_step=estimated_finish,
            deadline_step=state.deadline_step,
        )

    def _distance_latency_s(self, position: tuple[float, float], execution_rsu: str) -> float:
        if self.distance_latency_max_s <= 0.0:
            return 0.0

        # Normalize distance into [0, 1] and interpolate base..max latency.
        distance_m = self.rsu_registry.distance_to_rsu(position, execution_rsu)
        normalized = min(1.0, distance_m / self.distance_latency_norm_m)
        return self.distance_latency_base_s + (self.distance_latency_max_s - self.distance_latency_base_s) * normalized
